import carla
import erdos
import pickle
import threading

from pylot.perception.messages import PointCloudMessage
from pylot.perception.point_cloud import PointCloud
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode


class CarlaLidarDriverOperator(erdos.Operator):
    """CarlaLidarDriverOperator publishes Lidar point clouds onto a stream.

    This operator attaches a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the point clouds and
    publishes it to downstream operators.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the id of the ego vehicle. It uses this
            id to get a Carla handle to the vehicle.
        lidar_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends point cloud messages.
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the lidar sensor.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ground_vehicle_id_stream, release_sensor_stream,
                 lidar_stream, notify_reading_stream, lidar_setup, flags):
        erdos.add_watermark_callback([release_sensor_stream], [],
                                     self.release_data)
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._lidar_stream = lidar_stream
        self._notify_reading_stream = notify_reading_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._lidar_setup = lidar_setup
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        self._pickle_lock = threading.Lock()
        self._pickled_messages = {}
        # Handle to the Lidar Carla actor.
        self._lidar = None
        self._lock = threading.Lock()
        # If false then the operator does not send data until it receives
        # release data watermark. Otherwise, it sends as soon as it
        # receives it.
        self._release_data = False

    @staticmethod
    def connect(ground_vehicle_id_stream, release_sensor_stream):
        lidar_stream = erdos.WriteStream()
        notify_reading_stream = erdos.WriteStream()
        return [lidar_stream, notify_reading_stream]

    def release_data(self, timestamp):
        if timestamp.is_top:
            self._release_data = True
        else:
            watermark_msg = erdos.WatermarkMessage(timestamp)
            self._lidar_stream.send_pickled(timestamp,
                                            self._pickled_messages[timestamp])
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self._lidar_stream.send(watermark_msg)
            with self._pickle_lock:
                del self._pickled_messages[timestamp]

    def process_point_clouds(self, carla_pc):
        """ Invoked when a pointcloud is received from the simulator.

        Args:
            carla_pc: a carla.SensorData object.
        """
        game_time = int(carla_pc.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        watermark_msg = erdos.WatermarkMessage(timestamp)
        with erdos.profile(self.config.name + '.process_point_clouds',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            # Ensure that the code executes serially
            with self._lock:
                assert len(
                    carla_pc.raw_data) > 0, 'Lidar did not send any points'
                # Include the transform relative to the vehicle.
                # Carla carla_pc.transform returns the world transform, but
                # we do not use it directly.
                msg = PointCloudMessage(
                    timestamp,
                    PointCloud.from_carla_point_cloud(carla_pc,
                                                      self._lidar_setup))

                if self._release_data:
                    self._lidar_stream.send(msg)
                    self._lidar_stream.send(watermark_msg)
                else:
                    pickled_msg = pickle.dumps(
                        msg, protocol=pickle.HIGHEST_PROTOCOL)
                    with self._pickle_lock:
                        self._pickled_messages[msg.timestamp] = pickled_msg
                    self._notify_reading_stream.send(watermark_msg)

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The CarlaLidarDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        set_simulation_mode(world, self._flags)

        self._vehicle = get_vehicle_handle(world, vehicle_id)

        # Install the Lidar.
        lidar_blueprint = world.get_blueprint_library().find(
            self._lidar_setup.lidar_type)
        lidar_blueprint.set_attribute('channels',
                                      str(self._lidar_setup.channels))
        if (self._flags.carla_version == '0.9.7'
                or self._flags.carla_version == '0.9.8'):
            lidar_blueprint.set_attribute(
                'range', str(self._lidar_setup.get_range_in_meters()))
        else:
            lidar_blueprint.set_attribute('range',
                                          str(self._lidar_setup.range))
        lidar_blueprint.set_attribute('points_per_second',
                                      str(self._lidar_setup.points_per_second))
        lidar_blueprint.set_attribute(
            'rotation_frequency', str(self._lidar_setup.rotation_frequency))
        lidar_blueprint.set_attribute('upper_fov',
                                      str(self._lidar_setup.upper_fov))
        lidar_blueprint.set_attribute('lower_fov',
                                      str(self._lidar_setup.lower_fov))
        if self._flags.carla_lidar_frequency == -1:
            lidar_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            lidar_blueprint.set_attribute(
                'sensor_tick', str(1.0 / self._flags.carla_lidar_frequency))

        transform = self._lidar_setup.get_transform().as_carla_transform()

        self._logger.debug("Spawning a lidar: {}".format(self._lidar_setup))

        if self._flags.carla_version == '0.9.8':
            # Must attach lidar with a SpringArm, otherwise the point cloud is
            # empty.
            self._lidar = world.spawn_actor(
                lidar_blueprint,
                transform,
                attach_to=self._vehicle,
                attachment_type=carla.AttachmentType.SpringArm)
        else:
            self._lidar = world.spawn_actor(lidar_blueprint,
                                            transform,
                                            attach_to=self._vehicle)

        # Register the callback on the Lidar.
        self._lidar.listen(self.process_point_clouds)

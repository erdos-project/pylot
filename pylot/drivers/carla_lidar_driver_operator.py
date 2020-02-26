import erdos
import threading
import time

from pylot.perception.messages import PointCloudMessage
from pylot.perception.point_cloud import PointCloud
from pylot.simulation.utils import get_world, set_synchronous_mode


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
        name (:obj:`str`): The name of the operator.
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the lidar sensor.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 ground_vehicle_id_stream,
                 lidar_stream,
                 name,
                 lidar_setup,
                 flags,
                 log_file_name=None):
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._lidar_stream = lidar_stream
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._lidar_setup = lidar_setup
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        # Handle to the Lidar Carla actor.
        self._lidar = None
        self._lock = threading.Lock()

    @staticmethod
    def connect(ground_vehicle_id_stream):
        lidar_stream = erdos.WriteStream()
        return [lidar_stream]

    def process_point_clouds(self, carla_pc):
        """ Invoked when a pointcloud is received from the simulator.

        Args:
            carla_pc: a carla.SensorData object.
        """
        # Ensure that the code executes serially
        with self._lock:
            game_time = int(carla_pc.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            # Include the transform relative to the vehicle.
            # Carla carla_pc.transform returns the world transform, but
            # we do not use it directly.
            msg = PointCloudMessage(
                timestamp,
                PointCloud.from_carla_point_cloud(
                    carla_pc, self._lidar_setup.get_transform()))

            self._lidar_stream.send(msg)
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self._lidar_stream.send(watermark_msg)

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
        if self._flags.carla_synchronous_mode:
            set_synchronous_mode(world, self._flags.carla_fps)

        num_tries = 0
        while self._vehicle is None and num_tries < 30:
            self._vehicle = world.get_actors().find(vehicle_id)
            self._logger.debug(
                "Could not find vehicle. Try {}".format(num_tries))
            time.sleep(1)
            num_tries += 1
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the Lidar.
        lidar_blueprint = world.get_blueprint_library().find(
            self._lidar_setup.lidar_type)

        lidar_blueprint.set_attribute('channels',
                                      str(self._lidar_setup.channels))
        if self._flags.carla_version == '0.9.7':
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
        # XXX(ionel): Set sensor tick.
        # lidar_blueprint.set_attribute('sensor_tick')

        transform = self._lidar_setup.get_transform().as_carla_transform()

        self._logger.debug("Spawning a lidar: {}".format(self._lidar_setup))

        self._lidar = world.spawn_actor(lidar_blueprint,
                                        transform,
                                        attach_to=self._vehicle)
        # Register the callback on the Lidar.
        self._lidar.listen(self.process_point_clouds)
        # TODO: We might have to loop here to keep hold of the thread so that
        # Carla callbacks are still invoked.
        # while True:
        #     time.sleep(0.01)

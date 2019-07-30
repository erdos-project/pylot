import copy
import numpy as np
import threading

import pylot.utils
from pylot.simulation.carla_utils import get_world, to_carla_transform,\
    set_synchronous_mode
from pylot.simulation.messages import PointCloudMessage

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging
from erdos.message import WatermarkMessage
from erdos.timestamp import Timestamp


class LidarDriverOperator(Op):
    """ LidarDriverOperator publishes Lidar point clouds onto a stream.

    This operator attaches a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the point clouds and
    publishes it to downstream operators.

    Attributes:
        _lidar_setup: A LidarSetup tuple.
        _lidar: Handle to the Lidar inside the simulation.
        _vehicle: Handle to the hero vehicle inside the simulation.
    """
    def __init__(self, name, lidar_setup, flags, log_file_name=None):
        """ Initializes the Lidar inside the simulation with the given
        parameters.

        Args:
            name: The unique name of the operator.
            lidar_setup: A LidarSetup tuple..
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(LidarDriverOperator, self).__init__(
            name, no_watermark_passthrough=True)
        # The operator does not pass watermarks by defaults.
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._lidar_setup = lidar_setup
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        # Handle to the Lidar Carla actor.
        self._lidar = None
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams, lidar_setup):
        """ Set up callback functions on the input streams and return the
        output stream that publishes the point clouds.

        Args:
            input_streams: The streams that this operator is connected to.
            lidar_setup: A LidarSetup tuple.
        """
        input_streams.filter(pylot.utils.is_ground_vehicle_id_stream)\
                     .add_callback(LidarDriverOperator.on_vehicle_id)
        return [pylot.utils.create_lidar_stream(lidar_setup)]

    def process_point_clouds(self, carla_pc):
        """ Invoked when a pointcloud is received from the simulator.

        Args:
            carla_pc: a carla.SensorData object.
        """
        # Ensure that the code executes serially
        with self._lock:
            game_time = int(carla_pc.timestamp * 1000)
            timestamp = Timestamp(coordinates=[game_time])
            watermark_msg = WatermarkMessage(timestamp)

            # Transform the raw_data into a point cloud.
            points = np.frombuffer(carla_pc.raw_data, dtype=np.dtype('f4'))
            points = copy.deepcopy(points)
            points = np.reshape(points, (int(points.shape[0] / 3), 3))

            # Include the transform relative to the vehicle.
            # Carla carla_pc.transform returns the world transform, but
            # we do not use it directly.
            msg = PointCloudMessage(
                points,
                self._lidar_setup.get_transform(),
                timestamp)

            self.get_output_stream(self._lidar_setup.name).send(msg)
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self.get_output_stream(self._lidar_setup.name).send(watermark_msg)

    def on_vehicle_id(self, msg):
        """ This function receives the identifier for the vehicle, retrieves
        the handler for the vehicle from the simulation and attaches the
        camera to it.

        Args:
            msg: The identifier for the vehicle to attach the camera to.
        """
        vehicle_id = msg.data
        self._logger.info(
            "The LidarDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.carla_host,
                             self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        if self._flags.carla_synchronous_mode:
            set_synchronous_mode(world, self._flags.carla_fps)

        self._vehicle = world.get_actors().find(vehicle_id)
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the Lidar.
        lidar_blueprint = world.get_blueprint_library().find(
            self._lidar_setup.lidar_type)

        lidar_blueprint.set_attribute('channels',
                                      str(self._lidar_setup.channels))
        lidar_blueprint.set_attribute('range',
                                      str(self._lidar_setup.range))
        lidar_blueprint.set_attribute('points_per_second',
                                      str(self._lidar_setup.points_per_second))
        lidar_blueprint.set_attribute(
            'rotation_frequency',
            str(self._lidar_setup.rotation_frequency))
        lidar_blueprint.set_attribute('upper_fov',
                                      str(self._lidar_setup.upper_fov))
        lidar_blueprint.set_attribute('lower_fov',
                                      str(self._lidar_setup.lower_fov))
        # XXX(ionel): Set sensor tick.
        # lidar_blueprint.set_attribute('sensor_tick')

        transform = to_carla_transform(self._lidar_setup.get_transform())

        self._logger.info("Spawning a lidar: {}".format(self._lidar_setup))

        self._lidar = world.spawn_actor(lidar_blueprint,
                                        transform,
                                        attach_to=self._vehicle)
        # Register the callback on the Lidar.
        self._lidar.listen(self.process_point_clouds)

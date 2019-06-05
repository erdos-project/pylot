import copy
import numpy as np
import threading

import carla

import pylot.utils
from pylot.simulation.carla_utils import get_world
from pylot.simulation.messages import PointCloudMessage
from pylot.simulation.utils import to_erdos_transform


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
        super(LidarDriverOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._lidar_setup = lidar_setup

        _, self._world = get_world(
            self._flags.carla_host, self._flags.carla_port)
        if self._world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        # Starts from 1 because the world ticks once before the drivers are
        # added.
        self._message_cnt = 1
        self._vehicle = None
        self._lidar = None
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams, lidar_setup):
        input_streams.filter(pylot.utils.is_ground_vehicle_id_stream)\
                     .add_callback(LidarDriverOperator.on_vehicle_id)
        return [pylot.utils.create_lidar_stream(lidar_setup)]

    def process_point_clouds(self, carla_pc):
        with self._lock:
            game_time = int(carla_pc.timestamp * 1000)
            timestamp = Timestamp(coordinates=[game_time, self._message_cnt])
            watermark_msg = WatermarkMessage(timestamp)
            self._message_cnt += 1

            points = np.frombuffer(carla_pc.raw_data, dtype=np.dtype('f4'))
            points = copy.deepcopy(points)
            points = np.reshape(points, (int(points.shape[0] / 3), 3))

            msg = PointCloudMessage(
                points,
                to_erdos_transform(carla_pc.transform),
                timestamp)

            self.get_output_stream(self._lidar_setup.name).send(msg)
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
            "The LidarDriverOperator received the vehicle identifier: {}".format(
                vehicle_id))

        self._vehicle = self._world.get_actors().find(vehicle_id)
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the Lidar.
        lidar_blueprint = self._world.get_blueprint_library().find(
                self._lidar_setup.type)

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
        # TODO(ionel): Set sensor tick.
        #        lidar_blueprint.set_attribute('sensor_tick')

        transform = carla.Transform(
            carla.Location(*self._lidar_setup.pos),
            carla.Rotation(pitch=0, yaw=0, roll=0),
        )

        self._logger.info("Spawning a lidar: {}".format(self._lidar_setup))

        self._lidar = self._world.spawn_actor(lidar_blueprint,
                                              transform,
                                              attach_to=self._vehicle)
        # Register the callback on the Lidar.
        self._lidar.listen(self.process_point_clouds)

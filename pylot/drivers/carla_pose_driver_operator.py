import threading
import erdos
from erdos import ReadStream, Timestamp, WriteStream

from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode
import pylot.utils


class MatchesOperator(erdos.Operator):
    def __init__(self, left_stream, right_stream):
        self._left_msgs = []
        self._right_msgs = []
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        left_stream.add_callback(self.on_left_stream)
        right_stream.add_callback(self.on_right_stream)
        left_stream.add_watermark_callback(
            lambda t: self._logger.debug(f"@{t}: got left watermark"))
        right_stream.add_watermark_callback(
            lambda t: self._logger.debug(f"@{t}: got right watermark"))

        erdos.add_watermark_callback([left_stream, right_stream], [],
                                     self.on_watermark)

    @staticmethod
    def connect(left_stream, right_stream):
        return []

    def on_left_stream(self, msg):
        self._logger.debug("got left msg")
        self._left_msgs.append(msg)

    def on_right_stream(self, msg):
        self._logger.debug("got right msg")
        self._right_msgs.append(msg)

    def on_watermark(self, t):
        left_msgs, self._left_msgs = self._left_msgs, []
        right_msgs, self._right_msgs = self._right_msgs, []
        length_matches = (len(left_msgs) == len(right_msgs))
        left_tuples = [(m.timestamp, m.data) for m in left_msgs]
        right_tuples = [(m.timestamp, m.data) for m in right_msgs]
        matches = length_matches and all(
            map(
                lambda x: x[0].timestamp == x[1].timestamp and x[0].data == x[
                    1].data, zip(left_msgs, right_msgs)))

        if matches:
            self._logger.debug(f"@{t}: left matches right")
        else:
            self._logger.warn(
                f"@{t}: left does not match right\n\tleft: {left_tuples}\n\tright: {right_tuples}"
            )


class CarlaPoseDriverOperator(erdos.Operator):
    """Sends pose information."""
    def __init__(self, vehicle_id_stream: ReadStream, pose_stream: WriteStream,
                 frequency: float, flags):
        # Save the streams.
        self._vehicle_id_stream = vehicle_id_stream
        self._pose_stream = pose_stream

        # Save the flags and initialize logging.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Save the setup, the vehicle and the sensor.
        self._frequency = frequency
        self._vehicle = None
        self._gnss = None  # Use GNSS sensor to tick at a regular interval.
        self._lock = threading.Lock()

    @staticmethod
    def connect(vehicle_id_stream: ReadStream):
        pose_stream = WriteStream()
        return [pose_stream]

    def send_pose(self, timestamp: Timestamp):
        """Sends pose information followed by a watermark."""
        vec_transform = pylot.utils.Transform.from_simulator_transform(
            self._vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_simulator_vector(
            self._vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector,
                                timestamp.coordinates[0])
        self._logger.debug(f"@{timestamp}: sending {pose}")
        self._pose_stream.send(erdos.Message(timestamp, pose))
        self._pose_stream.send(erdos.WatermarkMessage(timestamp))

    def process_gnss(self, gnss_msg):
        """Callback attached to a GNSS sensor to ensure that pose is sent at a
        fixed frequency."""
        game_time = int(gnss_msg.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        with erdos.profile(self.config.name + '.process_gnss',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            with self._lock:
                self.send_pose(timestamp)

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id = self._vehicle_id_stream.read().data
        self._logger.debug("{} received the vehicle id: {}".format(
            self.config.name, vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        # Retrieve the vehicle and install the GNSS sensor.
        self._vehicle = get_vehicle_handle(world, vehicle_id)
        gnss_blueprint = world.get_blueprint_library().find(
            'sensor.other.gnss')

        if self._frequency == -1:
            gnss_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            gnss_blueprint.set_attribute('sensor_tick',
                                         str(1.0 / self._frequency))
        transform = pylot.utils.Transform(
            pylot.utils.Location(),
            pylot.utils.Rotation()).as_simulator_transform()
        self._logger.debug("Spawning a GNSS sensor")
        self._gnss = world.spawn_actor(gnss_blueprint,
                                       transform,
                                       attach_to=self._vehicle)

        # Register the callback on the GNSS sensor.
        self._gnss.listen(self.process_gnss)
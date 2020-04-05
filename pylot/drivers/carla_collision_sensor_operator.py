import carla
import erdos

from pylot.utils import Vector3D
from pylot.simulation.utils import get_world, get_vehicle_handle
from pylot.simulation.messages import CollisionMessage


class CarlaCollisionSensorDriverOperator(erdos.Operator):
    """ Publishes collision events of the ego-vehicle on a stream.

    This operator attaches a CollisionSensor to the ego-vehicle, registers
    callback functions to retrieve the collision events and publishes it to
    downstream operators.

    Note that more than one collision event may be published for a single
    timestamp, and it is advised to only work upon receiving watermarks from
    this operator.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the ID of the ego-vehicle.
        collision_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends the collision events.
        flags (absl.flags): Object to be used to access the absl flags.
    """

    def __init__(self, ground_vehicle_id_stream, collision_stream, flags):
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._collision_stream = collision_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        self._collision_sensor = None

        # Keep track of the last timestamp that we need to close.
        self._timestamp_to_close = None

    @staticmethod
    def connect(ground_vehicle_id_stream):
        collision_stream = erdos.WriteStream()
        return [collision_stream]

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug("@[{}]: Received Vehicle ID: {}".format(
            vehicle_id_msg.timestamp, vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        self._vehicle = get_vehicle_handle(world, vehicle_id)

        # Install the collision sensor.
        collision_blueprint = world.get_blueprint_library().find(
            'sensor.other.collision')
        self._logger.debug("Spawning a collision sensor.")
        self._collision_sensor = world.spawn_actor(collision_blueprint,
                                                   carla.Transform(),
                                                   attach_to=self._vehicle)

        # Register the callback on the collision sensor.
        self._collision_sensor.listen(self.process_collision)

        # Register an on_tick function to flow watermarks.
        world.on_tick(self.process_timestamp)

    def process_collision(self, collision_event):
        """ Invoked when a collision event is received from the simulation.

        Args:
            collision_event (:py:class:`carla.CollisionEvent`): A collision
                event that contains the impulse, location and the object with
                which the ego-vehicle collided.
        """
        game_time = int(collision_event.timestamp * 1000)
        self._logger.debug(
            "@[{}]: Received a collision event from the simulator.".format(
                game_time))

        # Create a CollisionMessage.
        timestamp = erdos.Timestamp(coordinates=[game_time])
        msg = CollisionMessage(
            collision_event.other_actor,
            Vector3D.from_carla_vector(collision_event.normal_impulse),
            timestamp)

        # Send the CollisionMessage.
        self._collision_stream.send(msg)

    def process_timestamp(self, msg):
        """ Invoked upon each tick of the simulator. This callback is used to
        flow watermarks to the downstream operators.

        Since multiple collision events can be published by Carla for a single
        timestamp, we need to wait till the next tick of the simulator to be
        sure that all the collision events for the previous timestamp have
        been sent to downstream operators.

        Args:
            msg (:py:class:`carla.WorldSettings`): A snapshot of the world at
            the given tick. We use this to retrieve the timestamp of the
            simulator.
        """
        timestamp = erdos.Timestamp(
            coordinates=[int(msg.elapsed_seconds * 1000)])
        if self._timestamp_to_close:
            last_timestamp = self._timestamp_to_close
            self._collision_stream.send(erdos.WatermarkMessage(last_timestamp))
        self._timestamp_to_close = timestamp

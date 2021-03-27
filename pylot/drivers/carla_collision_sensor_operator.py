"""This module implements an operator that publishes collision events
between the ego vehicle and other simulation agents.
"""

import erdos

from pylot.simulation.messages import CollisionMessage
from pylot.simulation.utils import get_vehicle_handle, get_world
from pylot.utils import Location, Rotation, Transform, Vector3D


class CarlaCollisionSensorDriverOperator(erdos.Operator):
    """Publishes collision events of the ego-vehicle on a stream.

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
    def __init__(self, ground_vehicle_id_stream: erdos.ReadStream,
                 collision_stream: erdos.WriteStream, flags):
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._collision_stream = collision_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # The hero vehicle actor object we obtain from the simulator.
        self._vehicle = None
        self._collision_sensor = None

    @staticmethod
    def connect(ground_vehicle_id_stream: erdos.ReadStream):
        collision_stream = erdos.WriteStream()
        return [collision_stream]

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug("@{}: Received Vehicle ID: {}".format(
            vehicle_id_msg.timestamp, vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)

        self._vehicle = get_vehicle_handle(world, vehicle_id)

        # Install the collision sensor.
        collision_blueprint = world.get_blueprint_library().find(
            'sensor.other.collision')
        self._logger.debug("Spawning a collision sensor.")
        self._collision_sensor = world.spawn_actor(
            collision_blueprint,
            Transform(Location(), Rotation()).as_simulator_transform(),
            attach_to=self._vehicle)

        # Register the callback on the collision sensor.
        self._collision_sensor.listen(self.process_collision)

    def process_collision(self, collision_event):
        """ Invoked when a collision event is received from the simulation.

        The collision event contains the impulse, location and the object with
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
            Vector3D.from_simulator_vector(collision_event.normal_impulse),
            timestamp)

        # Send the CollisionMessage.
        self._collision_stream.send(msg)
        # TODO(ionel): This code will fail if process_collision is called twice
        # for the same timestamp (i.e., if the vehicle collides with two other
        # actors)
        self._collision_stream.send(erdos.WatermarkMessage(timestamp))

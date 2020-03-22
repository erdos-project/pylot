import carla
import erdos
import math

from pylot.simulation.utils import get_vehicle_handle, get_world


class CarlaCollisionLoggerOperator(erdos.Operator):
    def __init__(self, vehicle_id_stream, can_bus_stream, flags):
        can_bus_stream.add_callback(self.on_can_bus_update)
        self._vehicle_id_stream = vehicle_id_stream
        self._flags = flags
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        self._collision_sensor = None
        self._ego_transform = None

    @staticmethod
    def connect(vehicle_id_stream, can_bus_stream):
        return []

    def on_can_bus_update(self, msg):
        self._ego_transform = msg.data.transform

    def on_collision(self, event):
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._csv_logger.info('collision,{},{},{},{}'.format(
            self._ego_transform.location, actor_type, impulse, intensity))

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        self._vehicle = get_vehicle_handle(world, vehicle_id)
        collision_bp = world.get_blueprint_library().find(
            'sensor.other.collision')
        self._collision_sensor = world.spawn_actor(collision_bp,
                                                   carla.Transform(),
                                                   attach_to=self._vehicle)
        self._collision_sensor.listen(self.on_collision)

from absl import flags

######################################################################
# Carla flags
######################################################################
flags.DEFINE_enum('carla_version', '0.9.6',
                  ['0.9.5', '0.9.6', '0.9.7', '0.9.8', '0.9.9'],
                  'Carla simulator version')
flags.DEFINE_string('carla_host', 'localhost', 'Carla host.')
flags.DEFINE_integer('carla_port', 2000, 'Carla port.')
flags.DEFINE_integer('carla_timeout', 10,
                     'Timeout for connecting to the Carla simulator.')
flags.DEFINE_enum('carla_mode', 'synchronous', [
    'synchronous', 'asynchronous', 'asynchronous-fixed-time-step',
    'pseudo-asynchronous'
], 'Sets the way in which to run the simulator')
flags.DEFINE_bool('carla_scenario_runner', False,
                  'True to enable running a scenario.')
flags.DEFINE_integer('carla_town', 1, 'Sets which Carla town to use')
flags.DEFINE_integer('carla_fps',
                     10,
                     'Carla simulator FPS; do not set below 10',
                     lower_bound=10)
flags.DEFINE_integer(
    'carla_camera_frequency', -1,
    'Sets the simulation time frequency at which frames are published. '
    '-1 means that a frame is published for each simulation tick')
flags.DEFINE_integer(
    'carla_imu_frequency', -1,
    'Sets the simulation time frequency at which IMUs are published. '
    '-1 means that a IMU message is published for each simulation tick')
flags.DEFINE_integer(
    'carla_lidar_frequency', -1,
    'Sets the simulation time frequency at which point clouds are published. '
    '-1 means that a point cloud is published for each simulation tick')
flags.DEFINE_integer(
    'carla_localization_frequency', -1,
    'Sets the simulation time at which ego-vehicle localization info is '
    'published. -1 means that info is published on each simulation tick')
flags.DEFINE_integer(
    'carla_control_frequency', -1,
    'Sets the simulation time at which ego-vehicle localization info is '
    'published for the control module. -1 means that info is published '
    'on each simulation tick')
flags.DEFINE_integer('carla_num_vehicles', 20, 'Carla num vehicles.')
flags.DEFINE_integer('carla_num_people', 40, 'Carla num people.')
flags.DEFINE_enum('carla_weather', 'ClearNoon', [
    'ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 'HardRainNoon',
    'HardRainSunset', 'MidRainSunset', 'MidRainyNoon', 'SoftRainNoon',
    'SoftRainSunset', 'WetCloudyNoon', 'WetCloudySunset', 'WetNoon',
    'WetSunset'
], 'Carla Weather Presets')
flags.DEFINE_integer(
    'carla_spawn_point_index', -1,
    'Index of spawn point where to place ego vehicle. -1 to randomly assign.')
flags.DEFINE_integer('carla_vehicle_mass', None,
                     'Modifies the mass of the ego-vehicle')
flags.DEFINE_float('carla_vehicle_moi', None,
                   'Modifies the moment of inertia of the ego-vehicle')

# Other flags
flags.DEFINE_integer('random_seed', None,
                     'Random seed for populating the simulation.')
flags.DEFINE_integer(
    'perfect_detection_max_distance', 125,
    'Limit perfect detection to a distance of this amount of meters')


def sensor_frequency_validator(flags_dict):
    return (
        flags_dict['carla_camera_frequency'] <= flags_dict['carla_fps']
        and flags_dict['carla_lidar_frequency'] <= flags_dict['carla_fps']
        and flags_dict['carla_imu_frequency'] <= flags_dict['carla_fps'] and
        flags_dict['carla_localization_frequency'] <= flags_dict['carla_fps']
        and flags_dict['carla_control_frequency'] <= flags_dict['carla_fps'])


flags.register_multi_flags_validator(
    [
        'carla_fps',
        'carla_camera_frequency',
        'carla_imu_frequency',
        'carla_lidar_frequency',
        'carla_localization_frequency',
        'carla_control_frequency',
    ],
    sensor_frequency_validator,
    message='Sensor frequencies cannot be greater than --carla_fps')

from absl import flags

######################################################################
# Simulator flags
######################################################################
flags.DEFINE_string('simulator_host', 'localhost', 'Simulator host.')
flags.DEFINE_integer('simulator_port', 2000, 'Simulator port.')
flags.DEFINE_integer('carla_traffic_manager_port', 8000,
                     'CARLA traffic manager port.')
flags.DEFINE_integer('simulator_timeout', 10,
                     'Timeout for connecting to the simulator.')
flags.DEFINE_enum('simulator_mode', 'synchronous', [
    'synchronous', 'asynchronous', 'asynchronous-fixed-time-step',
    'pseudo-asynchronous'
], 'Sets the way in which to run the simulator')
flags.DEFINE_bool('scenario_runner', False,
                  'Set to True to execute pylot with scenario runner.')
flags.DEFINE_integer('simulator_town', 1,
                     'Sets which simulator town to use. Options [1..7]')
flags.DEFINE_integer('simulator_fps',
                     20,
                     'Simulator FPS; do not set below 10',
                     lower_bound=10)
flags.DEFINE_integer(
    'simulator_camera_frequency', -1,
    'Sets the simulation time frequency at which frames are published. '
    '-1 means that a frame is published for each simulation tick')
flags.DEFINE_integer(
    'simulator_imu_frequency', -1,
    'Sets the simulation time frequency at which IMUs are published. '
    '-1 means that a IMU message is published for each simulation tick')
flags.DEFINE_integer(
    'simulator_gnss_frequency', -1,
    'Sets the simulation time frequency at which GNSS is published. '
    '-1 means that a GNSS message is published for each simulation tick')
flags.DEFINE_integer(
    'simulator_lidar_frequency', -1,
    'Sets the simulation time frequency at which point clouds are published. '
    '-1 means that a point cloud is published for each simulation tick')
flags.DEFINE_integer(
    'simulator_localization_frequency', -1,
    'Sets the simulation time at which ego-vehicle localization info is '
    'published. -1 means that info is published on each simulation tick')
flags.DEFINE_integer(
    'simulator_control_frequency', -1,
    'Sets the simulation time at which ego-vehicle localization info is '
    'published for the control module. -1 means that info is published '
    'on each simulation tick')
flags.DEFINE_integer('simulator_num_vehicles', 20,
                     'Sets the number of vehicles in the simulation.')
flags.DEFINE_integer('simulator_num_people', 40,
                     'Sets the number of people in the simulation.')
flags.DEFINE_enum('simulator_weather', 'ClearNoon', [
    'ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 'HardRainNoon',
    'HardRainSunset', 'MidRainSunset', 'MidRainyNoon', 'SoftRainNoon',
    'SoftRainSunset', 'WetCloudyNoon', 'WetCloudySunset', 'WetNoon',
    'WetSunset'
], 'Simulator weather presets')
flags.DEFINE_integer(
    'simulator_spawn_point_index', -1,
    'Index of spawn point where to place ego vehicle. -1 to randomly assign.')
flags.DEFINE_integer('simulator_vehicle_mass', None,
                     'Modifies the mass of the ego-vehicle')
flags.DEFINE_float('simulator_vehicle_moi', None,
                   'Modifies the moment of inertia of the ego-vehicle')

# Other flags
flags.DEFINE_integer('random_seed', None,
                     'Random seed for populating the simulation.')

# Noise and bias flags for IMU and GNSS.
flags.DEFINE_float('accel_noise_stddev_x', 0.0,
                   'Sets the noise on the X-axis of the accelerometer.')
flags.DEFINE_float('accel_noise_stddev_y', 0.0,
                   'Sets the noise on the Y-axis of the accelerometer.')
flags.DEFINE_float('accel_noise_stddev_z', 0.0,
                   'Sets the noise on the Z-axis of the accelerometer.')

flags.DEFINE_float('gyro_noise_stddev_x', 0.0,
                   'Sets the noise on the X-axis of the gyroscope.')
flags.DEFINE_float('gyro_noise_stddev_y', 0.0,
                   'Sets the noise on the Y-axis of the gyroscope.')
flags.DEFINE_float('gyro_noise_stddev_z', 0.0,
                   'Sets the noise on the Z-axis of the gyroscope.')

flags.DEFINE_float('gnss_noise_stddev_alt', 0.0,
                   'Sets the noise on the altitude of the GNSS sensor.')
flags.DEFINE_float('gnss_noise_stddev_lat', 0.0,
                   'Sets the noise on the latitude of the GNSS sensor.')
flags.DEFINE_float('gnss_noise_stddev_lon', 0.0,
                   'Sets the noise on the longitude of the GNSS sensor.')

flags.DEFINE_float('gnss_bias_alt', 0.0,
                   'Sets the bias on the altitude of the GNSS sensor.')
flags.DEFINE_float('gnss_bias_lat', 0.0,
                   'Sets the bias on the latitude of the GNSS sensor.')
flags.DEFINE_float('gnss_bias_lon', 0.0,
                   'Sets the bias on the longitude of the GNSS sensor.')


def sensor_frequency_validator(flags_dict):
    return flags_dict['simulator_camera_frequency'] <= \
        flags_dict['simulator_fps'] and \
        flags_dict['simulator_lidar_frequency'] <= \
        flags_dict['simulator_fps'] and \
        flags_dict['simulator_imu_frequency'] <= \
        flags_dict['simulator_fps'] and \
        flags_dict['simulator_localization_frequency'] <= \
        flags_dict['simulator_fps'] and \
        flags_dict['simulator_control_frequency'] <= \
        flags_dict['simulator_fps']


flags.register_multi_flags_validator(
    [
        'simulator_fps',
        'simulator_camera_frequency',
        'simulator_imu_frequency',
        'simulator_lidar_frequency',
        'simulator_localization_frequency',
        'simulator_control_frequency',
    ],
    sensor_frequency_validator,
    message='Sensor frequencies cannot be greater than --simulator_fps')

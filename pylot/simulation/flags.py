from absl import flags

######################################################################
# Carla flags
######################################################################
flags.DEFINE_enum('carla_version', '0.9.6', ['0.9.5', '0.9.6', '0.9.7'],
                  'Carla simulator version')
flags.DEFINE_string('carla_host', 'localhost', 'Carla host.')
flags.DEFINE_integer('carla_port', 2000, 'Carla port.')
flags.DEFINE_integer('carla_timeout', 10,
                     'Timeout for connecting to the Carla simulator.')
flags.DEFINE_bool('carla_synchronous_mode', True,
                  'Run Carla in synchronous mode.')
flags.DEFINE_bool('carla_scenario_runner', False,
                  'True to enable running a scenario.')
flags.DEFINE_integer('carla_town', 1, 'Sets which Carla town to use.')
flags.DEFINE_integer('carla_fps', 10,
                     'Carla simulator FPS; do not set bellow 10.')
flags.DEFINE_float(
    'carla_step_frequency', -1,
    'Target frequency of sending control commands. -1 if '
    'commands should be applied as fast as possible.')
flags.DEFINE_integer('carla_num_vehicles', 20, 'Carla num vehicles.')
flags.DEFINE_integer('carla_num_people', 40, 'Carla num people.')
flags.DEFINE_string('carla_weather', 'ClearNoon', 'Carla Weather Presets')
flags.DEFINE_integer(
    'carla_spawn_point_index', -1,
    'Index of spawn point where to place ego vehicle. -1 to randomly assign.')
flags.DEFINE_integer('carla_camera_image_width', 1920,
                     'Carla camera image width')
flags.DEFINE_integer('carla_camera_image_height', 1080,
                     'Carla camera image height')

flags.DEFINE_integer(
    'perfect_tracking_num_steps', None,
    'Limit on number of past steps returned by the perfect object tracker.')

# Other flags
flags.DEFINE_integer(
    'top_down_lateral_view', 20,
    'Distance in meters to the left and right of the '
    'ego-vehicle that the top-down camera shows.')

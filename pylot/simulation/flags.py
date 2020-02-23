from absl import flags

######################################################################
# Carla flags
######################################################################
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
flags.DEFINE_integer(
    'random_seed', None,
    'Random seed for populating the simulation.'
)

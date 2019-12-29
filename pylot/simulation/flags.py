from absl import flags

flags.DEFINE_integer(
    'perfect_tracking_num_steps', None,
    'Limit on number of past steps returned by the perfect object tracker.')

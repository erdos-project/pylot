from absl import flags

flags.DEFINE_integer(
    'prediction_num_past_steps', None,
    'Number of past steps of each agent given to the prediction module.')
flags.DEFINE_integer(
    'prediction_num_future_steps', None,
    'Number of future steps outputted by the prediction module.')

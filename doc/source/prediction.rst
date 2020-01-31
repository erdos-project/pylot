Prediction
==========

The package provides operators and classes useful for prediction future
trajectories of obstacles and other actors.

Execute the following command to run a prediction demo:

.. code-block:: bash

    python3 pylot.py --flagfile=configs/prediction.conf

Important flags:

- ``--prediction``: Enables the prediction component of the stack.
- ``--prediction_type``: Sets which prediction operator to use. Pylot currently
  offers only one prediction solution: a simple
  `linear predictor <pylot.prediction.html#module-pylot.prediction.linear\_predictor\_operator>`__.
- ``--prediction_num_past_steps``: Sets the number of past readings the
  prediction components uses. The duration of the history used for prediction
  is equal to the number of past steps multiplied by the time between each
  step run.
- ``--prediction_num_future_steps``: Sets the number of future steps to predict.
- ``--evaluate_prediction``: Enables computation and logging of accuracy metrics
  of the prediction component.
- ``--visualize_prediction``: Enables visualization of predicted trajectories.

from collections import deque
import numpy as np

from erdos.message import WatermarkMessage
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.prediction.messages import ObjPrediction, PredictionMessage
from pylot.simulation.utils import Location
import pylot.utils

class LinearPredictorOp(Op):
    """Operator that takes in past (x,y) locations of agents, and fits a linear model to these locations.
    """

    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """Initializes the LinearPredictor Operator."""
        super(LinearPredictorOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._output_stream_name = output_stream_name

        self._frame_cnt = 0

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        input_streams.filter(pylot.utils.is_ground_tracking_stream).add_callback(
            LinearPredictorOp.generate_predicted_trajectories)
        return [pylot.utils.create_linear_prediction_stream(output_stream_name)]

    def generate_predicted_trajectories(self, msg):
        self._logger.info('Timestamps {}'.format(msg.timestamp))
        self._frame_cnt += 1
        obj_predictions_list = []

        for obj in msg.obj_trajectories:
            # Time step matrices used in regression.
            num_steps = min(self._flags.prediction_num_past_steps, len(obj.trajectory))
            ts = np.zeros((num_steps, 2))
            future_ts = np.zeros((self._flags.prediction_num_future_steps, 2))
            for t in range(num_steps):
                ts[t][0] = -t
                ts[t][1] = 1
            for i in range(self._flags.prediction_num_future_steps):
                future_ts[i][0] = i + 1
                future_ts[i][1] = 1
        
            xy = np.zeros((num_steps, 2))
            for t in range(num_steps):
                location = obj.trajectory[-(t+1)] # t-th most recent step
                xy[t][0] = location.x
                xy[t][1] = location.y
            linear_model_params = np.linalg.lstsq(ts, xy, rcond=None)[0]
            # Predict future steps and convert to list of locations.
            predict_array = np.matmul(future_ts, linear_model_params)
            predictions = []
            for t in range(self._flags.prediction_num_future_steps):
                predictions.append(Location(x=predict_array[t][0], y=predict_array[t][1]))
            obj_predictions_list.append(ObjPrediction(obj.obj_class,
                                                      obj.obj_id,
                                                      1.0, # probability
                                                      predictions))
        self.get_output_stream(self._output_stream_name).send(
            PredictionMessage(msg.timestamp, obj_predictions_list)) 

    def execute(self):
        self.spin()


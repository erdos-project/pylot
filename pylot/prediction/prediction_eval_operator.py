from collections import deque
import threading

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.messages import ObjTrajectory, ObjTrajectoriesMessage
from pylot.prediction.messages import ObjPrediction, PredictionMessage
import pylot.utils

class PredictionEvalOperator(Op):
    """ Operator that calculates metrics for the quality of
        predicted trajectories."""

    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(PredictionEvalOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        # Message buffers.
        self._prediction_msgs = deque()
        self._tracking_msgs = deque()
        self._can_bus_msgs = deque()
        
        # Accumulated list of predictions, from oldest to newest.
        self._predictions = deque(maxlen=self._flags.prediction_num_future_steps)

        self._lock = threading.Lock()
        self._frame_cnt = 0

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_tracking_stream).add_callback(
            PredictionEvalOperator._on_tracking_update)
        input_streams.filter(pylot.utils.is_prediction_stream).add_callback(
            PredictionEvalOperator._on_prediction_update)
        # Register a callback on canbus data stream.
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PredictionEvalOperator._on_can_bus_update)
        # Register a watermark callback.
        input_streams.add_completion_callback(
            PredictionEvalOperator.on_notification)
        return []

    def _on_prediction_update(self, msg):
        with self._lock:
            self._prediction_msgs.append(msg)

    def _on_tracking_update(self, msg):
        with self._lock:
            self._tracking_msgs.append(msg)

    def _on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def synchronize_msg_buffers(self, timestamp, buffers):
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True


    def on_notification(self, msg):
        msg_buffers = [self._tracking_msgs, self._prediction_msgs, self._can_bus_msgs]
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    msg_buffers):
                return
            tracking_msg = self._tracking_msgs.popleft()
            prediction_msg = self._prediction_msgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()
        self._logger.info('Timestamps {} {} {}'.format(
            tracking_msg.timestamp, prediction_msg.timestamp, can_bus_msg.timestamp))
        assert (tracking_msg.timestamp == prediction_msg.timestamp ==
                can_bus_msg.timestamp)

        self._frame_cnt += 1
        can_bus_transform = can_bus_msg.data.transform

        # Start calculating metrics when we've taken sufficiently many steps.
        if len(self._predictions) == self._flags.prediction_num_future_steps:
            # Convert the tracking message to a dictionary with trajectories 
            # in world coordinates, for speedup when calculating metrics.
            ground_trajectories_dict = {}
            for obj in tracking_msg.obj_trajectories:
                cur_trajectory = []
                for location in obj.trajectory:
                    ego_coord = pylot.simulation.utils.Transform(
                        location=pylot.simulation.utils.Location(
                            x=location.x, y=location.y, z=location.z),
                        rotation=pylot.simulation.utils.Rotation())
                    world_coord = can_bus_transform * ego_coord
                    cur_trajectory.append(world_coord.location)

                ground_trajectories_dict[obj.obj_id] = \
                    ObjTrajectory(obj.obj_class, obj.obj_id, cur_trajectory)
            # Evaluate the prediction corresponding to the current set of ground truth past trajectories.
            self.calculate_metrics(ground_trajectories_dict,
                                   self._predictions[0].predictions)

        # Convert the prediction to world coordinates and append it to the queue.
        obj_predictions_list = []
        for obj in prediction_msg.predictions:
            cur_trajectory = []
            for location in obj.trajectory:
                ego_coord = pylot.simulation.utils.Transform(
                    location=pylot.simulation.utils.Location(
                        x=location.x, y=location.y, z=location.z),
                    rotation=pylot.simulation.utils.Rotation())
                world_coord = can_bus_transform * ego_coord
                cur_trajectory.append(world_coord.location)
            obj_predictions_list.append(ObjPrediction(obj.obj_class,
                                                      obj.id,
                                                      1.0, # probability
                                                      cur_trajectory))
        self._predictions.append(PredictionMessage(msg.timestamp, obj_predictions_list))

    def calculate_metrics(self, ground_trajectories, predictions):
        """ Calculates and logs MSD (mean squared distance), ADE (average
            displacement error), and FDE (final displacement error).

            Args:
                ground_trajectories: A dictionary of ground-truth past object trajectories.
                predictions: A PredictionMessage.
        """
        # Vehicle metrics.
        vehicle_cnt = 0
        vehicle_msd = 0.0
        vehicle_ade = 0.0
        vehicle_fde = 0.0

        # Pedestrian metrics.
        pedestrian_cnt = 0
        pedestrian_msd = 0.0
        pedestrian_ade = 0.0
        pedestrian_fde = 0.0

        for obj in predictions:
            predicted_trajectory = obj.trajectory
            ground_trajectory = ground_trajectories[obj.id].trajectory
            if obj.obj_class == 'vehicle':
                vehicle_cnt += 1
            elif obj.obj_class == 'pedestrian':
                pedestrian_cnt += 1
            else:
                raise ValueError('Unexpected object class.')
            l2_distance = 0.0
            l1_distance = 0.0
            for idx in range(1, len(predicted_trajectory) + 1):
                # Calculate MSD
                l2_distance += predicted_trajectory[-idx].distance(
                                   ground_trajectory[-idx])
                # Calculate ADE
                l1_distance += predicted_trajectory[-idx].l1_distance(
                                   ground_trajectory[-idx])
            l2_distance /= len(predicted_trajectory)
            l1_distance /= len(predicted_trajectory)
            fde = predicted_trajectory[-1].l1_distance(ground_trajectory[-1]) 
            if obj.obj_class == 'vehicle':
                vehicle_msd += l2_distance
                vehicle_ade += l1_distance
                vehicle_fde += fde
            elif obj.obj_class == 'pedestrian':
                pedestrian_msd += l2_distance
                pedestrian_ade += l1_distance
                pedestrian_fde += fde
            else:
                raise ValueError('Unexpected object class.')

        vehicle_msd /= vehicle_cnt
        vehicle_ade /= vehicle_cnt
        vehicle_fde /= vehicle_cnt
        pedestrian_msd /= pedestrian_cnt
        pedestrian_ade /= pedestrian_cnt
        pedestrian_fde /= pedestrian_cnt
        # Log metrics.
        self._logger.info('Vehicle MSD is: {}'.format(vehicle_msd))
        self._logger.info('Vehicle ADE is: {}'.format(vehicle_ade))
        self._logger.info('Vehicle FDE is: {}'.format(vehicle_fde))
        self._logger.info('Pedestrian MSD is: {}'.format(pedestrian_msd))
        self._logger.info('Pedestrian ADE is: {}'.format(pedestrian_ade))
        self._logger.info('Pedestrian FDE is: {}'.format(pedestrian_fde))

        self._csv_logger.info('{},{},{},{}'.format(
           time_epoch_ms(), self.name, 'vehicle-MSD', vehicle_msd))
        self._csv_logger.info('{},{},{},{}'.format(
           time_epoch_ms(), self.name, 'vehicle-ADE', vehicle_ade))
        self._csv_logger.info('{},{},{},{}'.format(
           time_epoch_ms(), self.name, 'vehicle-FDE', vehicle_fde))
        self._csv_logger.info('{},{},{},{}'.format(
           time_epoch_ms(), self.name, 'pedestrian-MSD', pedestrian_msd))
        self._csv_logger.info('{},{},{},{}'.format(
           time_epoch_ms(), self.name, 'pedestrian-ADE', pedestrian_ade))
        self._csv_logger.info('{},{},{},{}'.format(
           time_epoch_ms(), self.name, 'pedestrian-FDE', pedestrian_fde))


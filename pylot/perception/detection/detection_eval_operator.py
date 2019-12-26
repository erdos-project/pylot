from absl import flags
import erdust
import heapq

from pylot.perception.detection.utils import get_mAP
from pylot.utils import time_epoch_ms

flags.DEFINE_enum('detection_metric', 'mAP', ['mAP', 'timely-mAP'],
                  'Detection evaluation metric')


class DetectionEvalOperator(erdust.Operator):
    def __init__(self,
                 obstacles_stream,
                 ground_obstacles_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        obstacles_stream.add_callback(self.on_obstacles)
        ground_obstacles_stream.add_callback(self.on_ground_obstacles)
        erdust.add_watermark_callback(
            [obstacles_stream, ground_obstacles_stream], [],
            self.on_notification)
        self._name = name
        self._flags = flags
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._last_notification = None
        # Buffer of detected obstacles.
        self._detected_obstacles = []
        # Buffer of ground truth bboxes.
        self._ground_obstacles = []
        # Heap storing pairs of (ground/output time, game time).
        self._detector_start_end_times = []
        self._sim_interval = None

    @staticmethod
    def connect(obstacles_stream, ground_obstacles_stream):
        return []

    def on_notification(self, timestamp):
        game_time = timestamp.coordinates[0]
        if not self._last_notification:
            self._last_notification = game_time
            return
        else:
            self._sim_interval = (game_time - self._last_notification)
            self._last_notification = game_time

        while len(self._detector_start_end_times) > 0:
            (end_time, start_time) = self._detector_start_end_times[0]
            # We can compute mAP if the endtime is not greater than the ground
            # time.
            if end_time <= game_time:
                # This is the closest ground bounding box to the end time.
                heapq.heappop(self._detector_start_end_times)
                end_bboxes = self.__get_ground_obstacles_at(end_time)
                # Get detector output obstacles.
                det_objs = self.__get_obstacles_at(start_time)
                if (len(det_objs) > 0 or len(end_bboxes) > 0):
                    mAP = get_mAP(end_bboxes, det_objs)
                    self._logger.info('mAP is: {}'.format(mAP))
                    self._csv_logger.info('{},{},{},{}'.format(
                        time_epoch_ms(), self._name, 'mAP', mAP))
                self._logger.debug('Computing accuracy for {} {}'.format(
                    end_time, start_time))
            else:
                # The remaining entries require newer ground bboxes.
                break

        self.__garbage_collect_obstacles()

    def __get_ground_obstacles_at(self, timestamp):
        for (time, obstacles) in self._ground_obstacles:
            if time == timestamp:
                return obstacles
            elif time > timestamp:
                break
        self._logger.fatal(
            'Could not find ground obstacles for {}'.format(timestamp))

    def __get_obstacles_at(self, timestamp):
        for (time, obstacles) in self._detected_obstacles:
            if time == timestamp:
                return obstacles
            elif time > timestamp:
                break
        self._logger.fatal(
            'Could not find detected obstacles for {}'.format(timestamp))

    def __garbage_collect_obstacles(self):
        # Get the minimum watermark.
        watermark = None
        for (_, start_time) in self._detector_start_end_times:
            if watermark is None or start_time < watermark:
                watermark = start_time
        if watermark is None:
            return
        # Remove all detected obstacles that are below the watermark.
        index = 0
        while (index < len(self._detected_obstacles)
               and self._detected_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._detected_obstacles = self._detected_obstacles[index:]
        # Remove all the ground obstacles that are below the watermark.
        index = 0
        while (index < len(self._ground_obstacles)
               and self._ground_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._ground_obstacles = self._ground_obstacles[index:]

    def on_obstacles(self, msg):
        game_time = msg.timestamp.coordinates[0]
        vehicles_bboxes, ped_bboxes, _ = self.__get_bboxes_by_category(
            msg.detected_objects)
        self._detected_obstacles.append(
            (game_time, vehicles_bboxes + ped_bboxes))
        # Two metrics: 1) mAP, and 2) timely-mAP
        if self._flags.detection_metric == 'mAP':
            # We will compare the bboxes with the ground truth at the same
            # game time.
            heapq.heappush(self._detector_start_end_times,
                           (game_time, game_time))
        elif self._flags.detection_metric == 'timely-mAP':
            # Ground bboxes time should be as close as possible to the time of
            # the obstacles + detector runtime.
            ground_bboxes_time = self.__compute_closest_frame_time(game_time +
                                                                   msg.runtime)
            # Round time to nearest frame.
            heapq.heappush(self._detector_start_end_times,
                           (ground_bboxes_time, game_time))
        else:
            raise ValueError('Unexpected detection metric {}'.format(
                self._flags.detection_metric))

    def on_ground_obstacles(self, msg):
        game_time = msg.timestamp.coordinates[0]
        vehicles_bboxes, ped_bboxes, _ = self.__get_bboxes_by_category(
            msg.detected_objects)
        self._ground_obstacles.append(
            (game_time, ped_bboxes + vehicles_bboxes))

    def __compute_closest_frame_time(self, time):
        base = int(time) / self._sim_interval * self._sim_interval
        if time - base < self._sim_interval / 2:
            return base
        else:
            return base + self._sim_interval

    def __get_bboxes_by_category(self, det_objs):
        """ Divides perception.detection.utils.DetectedObject by labels."""
        vehicles = []
        pedestrians = []
        traffic_lights = []
        for det_obj in det_objs:
            if det_obj.label == 'vehicle':
                vehicles.append(det_obj.corners)
            elif det_obj.label == 'pedestrian':
                pedestrians.append(det_obj.corners)
            elif det_obj.label == 'traffic_light':
                traffic_lights.append(det_obj.corners)
            else:
                self._logger.warning('Unexpected label {}'.format(
                    det_obj.label))
        return vehicles, pedestrians, traffic_lights

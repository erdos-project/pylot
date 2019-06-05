import heapq

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.detection.utils import get_pedestrian_mAP, visualize_ground_bboxes
import pylot.utils
from pylot.simulation.utils import get_2d_bbox_from_3d_box, get_camera_intrinsic_and_transform, have_same_depth, map_ground_3D_transform_to_2D


class ObstacleAccuracyOperator(Op):

    def __init__(self,
                 name,
                 rgb_camera_setup,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(ObstacleAccuracyOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._vehicle_transforms = []
        self._pedestrians = []
        self._vehicles = []
        self._traffic_lights = []
        self._traffic_signs = []
        self._depth_imgs = []
        self._bgr_imgs = []
        (camera_name, _, img_size, pos) = rgb_camera_setup
        (self._rgb_intrinsic, self._rgb_transform, self._rgb_img_size) = get_camera_intrinsic_and_transform(
            image_size=img_size, position=pos)
        self._last_notification = -1
        # Buffer of detected obstacles.
        self._detected_obstacles = []
        # Buffer of ground truth bboxes.
        self._ground_obstacles = []
        # Heap storing pairs of (ground/output time, game time).
        self._detector_start_end_times = []
        self._sim_interval = None
        self._iou_thresholds = [0.1 * i for i in range(1, 10)]

    @staticmethod
    def setup_streams(input_streams, depth_camera_name):
        input_streams.filter_name(depth_camera_name).add_callback(
            ObstacleAccuracyOperator.on_depth_camera_update)
        input_streams.filter(pylot.utils.is_camera_stream).add_callback(
            ObstacleAccuracyOperator.on_bgr_camera_update)
        input_streams.filter(pylot.utils.is_vehicle_transform_stream).add_callback(
            ObstacleAccuracyOperator.on_vehicle_transform_update)
        input_streams.filter(pylot.utils.is_ground_pedestrians_stream).add_callback(
            ObstacleAccuracyOperator.on_pedestrians_update)
        input_streams.filter(pylot.utils.is_ground_vehicles_stream).add_callback(
            ObstacleAccuracyOperator.on_vehicles_update)
        input_streams.filter(pylot.utils.is_ground_traffic_lights_stream).add_callback(
            ObstacleAccuracyOperator.on_traffic_lights_update)
        input_streams.filter(pylot.utils.is_ground_traffic_signs_stream).add_callback(
            ObstacleAccuracyOperator.on_traffic_signs_update)
        input_streams.filter(pylot.utils.is_obstacles_stream).add_callback(
            ObstacleAccuracyOperator.on_obstacles)
        # Register a watermark callback.
        input_streams.add_completion_callback(
            ObstacleAccuracyOperator.on_notification)
        return []

    def on_notification(self, msg):
        # Check that we didn't skip any notification. We only skip
        # notifications if messages or watermarks are lost.
        if self._last_notification != -1:
            assert self._last_notification + 1 == msg.timestamp.coordinates[1]
        self._last_notification = msg.timestamp.coordinates[1]

        # Ignore the first two messages. We use them to get sim time
        # between frames.
        if self._last_notification < 2:
            if self._last_notification == 0:
                self._sim_interval = int(msg.timestamp.coordinates[0])
            elif self._last_notification == 1:
                # Set he real simulation interval.
                self._sim_interval = int(msg.timestamp.coordinates[0]) - self._sim_interval
            return

        game_time = msg.timestamp.coordinates[0]
        # Transform the 3D boxes at time watermark game time to 2D.
        (ped_bboxes, vec_bboxes) = self.__get_bboxes()
        # Add the pedestrians to the ground obstacles buffer.
        self._ground_obstacles.append((game_time, ped_bboxes))

        while len(self._detector_start_end_times) > 0:
            (end_time, start_time) = self._detector_start_end_times[0]
            # We can compute mAP if the endtime is not greater than the ground time.
            if end_time <= game_time:
                # This is the closest ground bounding box to the end time.
                heapq.heappop(self._detector_start_end_times)
                end_bboxes = self.__get_ground_obstacles_at(end_time)
                if self._flags.detection_eval_use_accuracy_model:
                    # Not using the detector's outputs => get ground bboxes.
                    start_bboxes = self.__get_ground_obstacles_at(start_time)
                    if (len(start_bboxes) > 0 or len(end_bboxes) > 0):
                        precisions = []
                        for iou in self._iou_thresholds:
                            (precision, _) = get_precision_recall_at_iou(
                                end_bboxes, start_bboxes, iou)
                            precisions.append(precision)
                        avg_precision = float(sum(precisions)) / len(precisions)
                        self._logger.info('precision-IoU is: {}'.format(avg_precision))
                        self._csv_logger.info('{},{},{},{}'.format(
                            time_epoch_ms(), self.name, 'precision-IoU', avg_precision))
                else:
                    # Get detector output obstacles.
                    det_objs = self.__get_obstacles_at(start_time)
                    if (len(det_objs) > 0 or len(end_bboxes) > 0):
                        mAP = get_pedestrian_mAP(end_bboxes, det_objs)
                        self._logger.info('mAP is: {}'.format(mAP))
                        self._csv_logger.info('{},{},{},{}'.format(
                            time_epoch_ms(), self.name, 'mAP', mAP))
                self._logger.info('Computing accuracy for {} {}'.format(
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
        while (index < len(self._detected_obstacles) and
               self._detected_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._detected_obstacles = self._detected_obstacles[index:]
        # Remove all the ground obstacles that are below the watermark.
        index = 0
        while (index < len(self._ground_obstacles) and
               self._ground_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._ground_obstacles = self._ground_obstacles[index:]

    def on_vehicle_transform_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._vehicle_transforms.append(msg)

    def on_pedestrians_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._pedestrians.append(msg)

    def on_vehicles_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._vehicles.append(msg)

    def on_traffic_lights_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._traffic_lights.append(msg)

    def on_traffic_signs_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._traffic_signs.append(msg)

    def on_depth_camera_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._depth_imgs.append(msg)

    def on_bgr_camera_update(self, msg):
        if msg.timestamp.coordinates[1] >= 2:
            self._bgr_imgs.append(msg)

    def on_obstacles(self, msg):
        # Ignore the first two messages. We use them to get sim time
        # between frames.
        if msg.timestamp.coordinates[1] < 2:
            return
        game_time = msg.timestamp.coordinates[0]
        self._detected_obstacles.append((game_time, msg.detected_objects))
        # Two metrics: 1) mAP, and 2) timely-mAP
        if self._flags.eval_detection_metric == 'mAP':
            # We will compare the bboxes with the ground truth at the same
            # game time.
            heapq.heappush(self._detector_start_end_times, (game_time, game_time))
        elif self._flags.eval_detection_metric == 'timely-mAP':
            # Ground bboxes time should be as close as possible to the time of the
            # obstacles + detector runtime.
            ground_bboxes_time = game_time + msg.runtime
            if self._flags.detection_eval_use_accuracy_model:
                # Include the decay of detection with time if we do not want to use
                # the accuracy of our models.
                # TODO(ionel): We must pass model mAP to this method.
                ground_bboxes_time += self.__mAP_to_latency(1)
            ground_bboxes_time = self.__compute_closest_frame_time(ground_bboxes_time)
            # Round time to nearest frame.
            heapq.heappush(self._detector_start_end_times,
                           (ground_bboxes_time, game_time))
        else:
            self._logger.fatal('Unexpected detection metric {}'.format(
                self._flags.eval_detection_metric))

    def execute(self):
        self.spin()

    def __compute_closest_frame_time(self, time):
        base = int(time) / self._sim_interval * self._sim_interval
        if time - base < self._sim_interval / 2:
            return base
        else:
            return base + self._sim_interval

    def __get_bboxes(self):
        self._logger.info("Timestamps {} {} {} {} {} {} {}".format(
            self._vehicle_transforms[0].timestamp,
            self._pedestrians[0].timestamp,
            self._vehicles[0].timestamp,
            self._traffic_lights[0].timestamp,
            self._traffic_signs[0].timestamp,
            self._depth_imgs[0].timestamp,
            self._bgr_imgs[0].timestamp))

        timestamp = self._pedestrians[0].timestamp
        vehicle_transform = self._vehicle_transforms[0].data
        self._vehicle_transforms = self._vehicle_transforms[1:]

        # Get the latest BGR and depth images.
        # NOTE: depth_to_array flips the image.
        depth_array = self._depth_imgs[0].data
        self._depth_imgs = self._depth_imgs[1:]

        bgr_img = self._bgr_imgs[0].data
        self._bgr_imgs = self._bgr_imgs[1:]

        # Get bboxes for pedestrians.
        pedestrians = self._pedestrians[0].pedestrians
        self._pedestrians = self._pedestrians[1:]
        ped_bboxes = self.__get_pedestrians_bboxes(
            pedestrians, vehicle_transform, depth_array)

        # Get bboxes for vehicles.
        vehicles = self._vehicles[0].vehicles
        self._vehicles = self._vehicles[1:]
        vec_bboxes = self.__get_vehicles_bboxes(
            vehicles, vehicle_transform, depth_array)

        # Get bboxes for traffic lights.
        traffic_lights = self._traffic_lights[0].traffic_lights
        self._traffic_lights = self._traffic_lights[1:]
        # self.__get_traffic_light_bboxes(traffic_lights, vehicle_transform,
        #                                 depth_array)

        # Get bboxes for the traffic signs.
        traffic_signs = self._traffic_signs[0].speed_signs
        self._traffic_signs = self._traffic_signs[1:]
        # self.__get_traffic_sign_bboxes(traffic_signs, vehicle_transform,
        #                                depth_array)

        if self._flags.visualize_ground_obstacles:
            visualize_ground_bboxes(
                self.name, timestamp, bgr_img, ped_bboxes, vec_bboxes)

        return (ped_bboxes, vec_bboxes)

    def __get_traffic_light_bboxes(self, traffic_lights, vehicle_transform,
                                   depth_array):
        tl_bboxes = []
        for tl in traffic_lights:
            pos = map_ground_3D_transform_to_2D(tl.location,
                                                vehicle_transform,
                                                self._rgb_transform,
                                                self._rgb_intrinsic,
                                                self._rgb_img_size)
            if pos is not None:
                x = int(pos[0])
                y = int(pos[1])
                z = pos[2].flatten().item(0)
                if have_same_depth(x, y, z, depth_array, 1.0):
                    # TODO(ionel): Figure out bounding box size.
                    tl_bboxes.append((x - 2, x + 2, y - 2, y + 2))
        return tl_bboxes

    def __get_traffic_sign_bboxes(self, traffic_signs, vehicle_transform,
                                  depth_array):
        ts_bboxes = []
        for traffic_sign in traffic_signs:
            pos = map_ground_3D_transform_to_2D(traffic_sign.location,
                                                vehicle_transform,
                                                self._rgb_transform,
                                                self._rgb_intrinsic,
                                                self._rgb_img_size)
            if pos is not None:
                x = int(pos[0])
                y = int(pos[1])
                z = pos[2].flatten().item(0)
                if have_same_depth(x, y, z, depth_array, 1.0):
                    # TODO(ionel): Figure out bounding box size.
                    ts_bboxes.append((x - 2, x + 2, y - 2, y + 2))
        return ts_bboxes

    def __get_pedestrians_bboxes(self, pedestrians, vehicle_transform,
                                 depth_array):
        ped_bboxes = []
        for pedestrian in pedestrians:
            bbox = get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, pedestrian.transform,
                pedestrian.bounding_box, self._rgb_transform, self._rgb_intrinsic,
                self._rgb_img_size, 1.5, 3.0)
            if bbox is not None:
                ped_bboxes.append(bbox)
        return ped_bboxes

    def __get_vehicles_bboxes(self, vehicles, vehicle_transform, depth_array):
        vec_bboxes = []
        for vehicle in vehicles:
            bbox = get_2d_bbox_from_3d_box(
                depth_array, vehicle_transform, vehicle.transform,
                vehicle.bounding_box, self._rgb_transform, self._rgb_intrinsic,
                self._rgb_img_size, 3.0, 3.0)
            if bbox is not None:
                vec_bboxes.append(bbox)
        return vec_bboxes

    def __mAP_to_latency(self, mAP):
        """ Function that gives a latency estimate of how much simulation
        time must pass for a perfect detector to decay to mAP.
        """
        # TODO(ionel): Implement!
        return 0

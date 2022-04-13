import time

import erdos
from erdos.context import OneInOneOutContext
from erdos.operator import OneInOneOut

from pylot.perception.camera_frame import CameraFrame

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS
from pylot.perception.messages import ObstaclesMessageTuple


class QdTrackOperator(OneInOneOut):
    def __init__(self, flags, camera_setup):
        from qdtrack.apis import init_model

        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._camera_setup = camera_setup
        self.model = init_model(self._flags.qd_track_config_path,
                                checkpoint=self._flags.qd_track_model_path,
                                device='cuda:0',
                                cfg_options=None)
        self.classes = ('pedestrian', 'rider', 'car', 'bus', 'truck',
                        'bicycle', 'motorcycle', 'train')
        self.frame_id = 0

    def on_data(self, context: OneInOneOutContext, data: CameraFrame):
        """Invoked when a FrameMessage is received on the camera stream."""
        from qdtrack.apis import inference_model

        self._logger.debug('@{}: {} received frame'.format(
            context.timestamp, self.config.name))
        assert data.encoding == 'BGR', 'Expects BGR frames'
        start_time = time.time()
        image_np = data.as_bgr_numpy_array()
        results = inference_model(self.model, image_np, self.frame_id)
        self.frame_id += 1

        bbox_result, track_result = results.values()
        obstacles = []
        for k, v in track_result.items():
            track_id = k
            bbox = v['bbox'][None, :]
            score = bbox[4]
            label_id = v['label']
            label = self.classes[label_id]
            if label in ['pedestrian', 'rider']:
                label = 'person'
            if label in OBSTACLE_LABELS:
                bounding_box_2D = BoundingBox2D(bbox[0], bbox[2], bbox[1],
                                                bbox[3])
                obstacles.append(
                    Obstacle(bounding_box_2D,
                             score,
                             label,
                             track_id,
                             bounding_box_2D=bounding_box_2D))
        runtime = (time.time() - start_time) * 1000
        context.write_stream.send(
            erdos.Message(context.timestamp,
                          ObstaclesMessageTuple(obstacles, runtime)))

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

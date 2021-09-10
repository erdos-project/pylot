import logging
import time

import erdos

import numpy as np
import PIL.Image as Image

from absl import flags, app

import pylot.utils
import pylot.flags
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage

import tensorflow as tf


FLAGS = flags.FLAGS

class DetectionModel():
    def __init__(self):
        self._detection_graph = tf.Graph()
        self._model_name = FLAGS.obstacle_detection_model_names[0]
        self._model_path = '../../../' + FLAGS.obstacle_detection_model_paths[0]
        self._flags = FLAGS

        print('Using Tensorflow: ', tf.__version__)
        print('GPU Available: ', tf.test.is_gpu_available())

        physical_devices = tf.config.experimental.list_physical_devices('GPU') 
        # Equivalent to allow_growth=True
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # Load the model from the model file.
        # pylot.utils.set_tf_loglevel(logging.ERROR)
        # with self._detection_graph.as_default():
        #     od_graph_def = tf.compat.v1.GraphDef()
        #     with tf.io.gfile.GFile(self._model_path, 'rb') as fid:
        #         serialized_graph = fid.read()
        #         od_graph_def.ParseFromString(serialized_graph)
        #         tf.import_graph_def(od_graph_def, name='')

        self._model = tf.saved_model.load('converted_model/')

        self._coco_labels = load_coco_labels('../../../' + self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)

        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0

    def process_image(self, image_np, width=192, height=108):
        start_time = time.time()
        
        num_detections, res_boxes, res_scores, res_classes = self.run_model(
            image_np)
        obstacles = []

        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >=
                        self._flags.obstacle_detection_min_score_threshold):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] *
                                    width),
                                int(res_boxes[i][3] *
                                    width),
                                int(res_boxes[i][0] *
                                    height),
                                int(res_boxes[i][2] *
                                    height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id))
                        self._unique_id += 1
                    else:
                        print('Ignoring non essential detection {}'.format(
                                self._coco_labels[res_classes[i]]))
            else:
                print('Filtering unknown class: {}'.format(
                    res_classes[i]))

        print('Took ' + str((time.time() - start_time) * 1000) + ' ms')
        print(obstacles)


    def run_model(self, image_np):
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return num_detections, res_boxes, res_scores, res_classes


def main(args):
    dm = DetectionModel()
    img = Image.open('../../../data/2021-09-09T18:35:04/center/center-6810.png')
    img = img.resize((192, 108))
    image_np = np.asarray(img)

    dm.process_image(image_np)

if __name__ == '__main__':
    app.run(main)


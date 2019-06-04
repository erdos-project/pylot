import pickle
import time

from erdos.data_stream import DataStream
from erdos.message import Message
from erdos.op import Op
from erdos.utils import setup_logging

from cv_bridge import CvBridge
import cv2
import pydarknet

from matplotlib import pyplot as plt


class DetectionOperator(Op):
    def __init__(self, name, output_stream_name):
        super(DetectionOperator, self).__init__(name)
        self._logger = setup_logging(self.name)
        self._output_stream_name = output_stream_name
        self._net = None
        self.bridge = None

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        input_streams.add_callback(DetectionOperator.on_msg_camera_stream)
        # TODO(Ionel): specify data type here
        return [DataStream(name=output_stream_name)]

    def add_bounding_boxes(self, cv_img, results):
        for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(
                cv_img, (int(x - w / 2), int(y - h / 2)),
                (int(x + w / 2), int(y + h / 2)), (255, 0, 0),
                thickness=2)
            cv2.putText(cv_img, str(cat.decode("utf-8")), (int(x), int(y)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
        plt.imshow(cv_img)
        plt.show()
        #cv2.imshow("output", cv_img)

    def on_msg_camera_stream(self, msg):
        self._logger.info('%s received frame %s', self.name, msg.timestamp)
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        cv_img = msg.frame
        img = pydarknet.Image(cv_img)
        results = self._net.detect(img)
        #self.add_bounding_boxes(cv_img, results)
        # bb_file = open('images/bounding_boxes{}'.format(msg.timestamp.coordinates[1]), 'wb')
        # pickle.dump(results, bb_file)
        # bb_file.close()
        # output_msg = Message((msg.data, results), msg.timestamp)

        runtime = (time.time() - start_time) * 1000
        self._logger.info('Object detector {} runtime {}'.format(
            self.name, runtime))

        output_msg = Message((results, runtime), msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)
        #self.notify_at(msg.timestamp)

    def on_notify(self, timestamp):
        self._logger.info('Received notification for %s', timestamp)

    def execute(self):
        self.bridge = CvBridge()
        self._net = pydarknet.Detector("dependencies/cfg/yolov3.cfg",
                                       "dependencies/data/yolov3.weights", 0,
                                       "dependencies/cfg/coco.data")
        self.spin()

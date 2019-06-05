import cv2
import dla.DLASeg
import numpy as np
import time
import torch

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.messages import SegmentedFrameMessage
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette
from pylot.utils import add_timestamp, create_segmented_camera_stream, is_camera_stream, rgb_to_bgr, bgra_to_bgr


class SegmentationDLAOperator(Op):
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(SegmentationDLAOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name
        # TODO(ionel): Figure out how to set GPU memory fraction.
        self._network = dla.DLASeg.DLASeg()
        self._network.load_state_dict(
            torch.load(self._flags.segmentation_dla_model_path))
        if self._flags.segmentation_gpu:
            self._network = self._network.cuda()

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        # Register a callback on the camera input stream.
        input_streams.filter(is_camera_stream).add_callback(
            SegmentationDLAOperator.on_msg_camera_stream)
        return [create_segmented_camera_stream(output_stream_name)]

    def on_msg_camera_stream(self, msg):
        self._logger.info('%s received frame %s', self.name, msg.timestamp)
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image = np.expand_dims(msg.frame.transpose([2, 0, 1]), axis=0)
        tensor = torch.tensor(image).float().cuda() / 255.0
        output = self._network(tensor)
        # XXX(ionel): Check if the model outputs Carla Cityscapes values or
        # correct Cityscapes values.
        output = transform_to_cityscapes_palette(
            torch.argmax(output, dim=1).cpu().numpy()[0])

        output = rgb_to_bgr(output)

        if self._flags.visualize_segmentation_output:
            add_timestamp(msg.timestamp, output)
            cv2.imshow(self.name, output)
            cv2.waitKey(1)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))

        output_msg = SegmentedFrameMessage(output, runtime, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

    def execute(self):
        self.spin()

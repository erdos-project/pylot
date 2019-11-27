import cv2
import drn.segment
from drn.segment import DRNSeg
import erdust
from torch.autograd import Variable
import time
import torch

from pylot.perception.messages import SegmentedFrameMessage
from pylot.utils import add_timestamp, rgb_to_bgr, time_epoch_ms


class SegmentationDRNOperator(erdust.Operator):
    """ Subscribes to a camera stream, and segments frames using DRN."""

    def __init__(self,
                 camera_stream,
                 segmented_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        camera_stream.add_callback(
            self.on_msg_camera_stream, [segmented_stream])
        self._name = name
        self._flags = flags
        self._logger = erdust.setup_logging(name, log_file_name)
        self._csv_logger = erdust.setup_csv_logging(
            name + '-csv', csv_file_name)
        arch = "drn_d_22"
        classes = 19
        self._pallete = drn.segment.CITYSCAPE_PALETTE
        # TODO(ionel): Figure out how to set GPU memory fraction.
        self._model = DRNSeg(
            arch, classes, pretrained_model=None, pretrained=False)
        self._model.load_state_dict(
            torch.load(self._flags.segmentation_drn_model_path))
        if torch.cuda.is_available():
            self._model = torch.nn.DataParallel(self._model).cuda()

    @staticmethod
    def connect(camera_stream):
        segmented_stream = erdust.WriteStream()
        return [segmented_stream]

    def on_msg_camera_stream(self, msg, segmented_stream):
        """Camera stream callback method.
        Invoked upon the receipt of a message on the camera stream.
        """
        self._logger.info('{} received frame {}'.format(
            self._name, msg.timestamp))
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image = torch.from_numpy(msg.frame.transpose([2, 0,
                                                      1])).unsqueeze(0).float()
        image_var = Variable(image, requires_grad=False, volatile=True)

        final = self._model(image_var)[0]
        _, pred = torch.max(final, 1)

        pred = pred.cpu().data.numpy()[0]
        image_np = self._pallete[pred.squeeze()]
        # After we apply the pallete, the image is in RGB format
        image_np = rgb_to_bgr(image_np)

        if self._flags.visualize_segmentation_output:
            add_timestamp(msg.timestamp, image_np)
            cv2.imshow(self._name, image_np)
            cv2.waitKey(1)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self._name, msg.timestamp, runtime))

        segmented_stream.send(
            SegmentedFrameMessage(image_np, runtime, msg.timestamp))

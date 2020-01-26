"""Implements an operator that semantically segments frames."""

from absl import flags
import drn.segment
from drn.segment import DRNSeg
import erdos
import time
from torch.autograd import Variable
import torch

from pylot.perception.messages import SegmentedFrameMessage
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.utils import time_epoch_ms

flags.DEFINE_string(
    'segmentation_model_path',
    'dependencies/models/segmentation/drn/drn_d_22_cityscapes.pth',
    'Path to the model')
flags.DEFINE_bool('visualize_segmentation_output', False,
                  'True to enable visualization of segmentation output')


class SegmentationDRNOperator(erdos.Operator):
    """Semantically segments frames using a DRN segmentation model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        segmented_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
            messages.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 camera_stream,
                 segmented_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [segmented_stream])
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        arch = "drn_d_22"
        classes = 19
        self._pallete = drn.segment.CARLA_CITYSCAPE_PALETTE
        self._model = DRNSeg(arch,
                             classes,
                             pretrained_model=None,
                             pretrained=False)
        self._model.load_state_dict(
            torch.load(self._flags.segmentation_model_path))
        if torch.cuda.is_available():
            self._model = torch.nn.DataParallel(self._model).cuda()

    @staticmethod
    def connect(camera_stream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
            messages.
        """
        segmented_stream = erdos.WriteStream()
        return [segmented_stream]

    def on_msg_camera_stream(self, msg, segmented_stream):
        """Invoked upon the receipt of a message on the camera stream.

        Args:
            msg: A :py:class:`~pylot.perception.messages.FrameMessage`.
            segmented_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
                messages.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        start_time = time.time()
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        image = torch.from_numpy(msg.frame.frame.transpose(
            [2, 0, 1])).unsqueeze(0).float()
        image_var = Variable(image, requires_grad=False, volatile=True)

        final = self._model(image_var)[0]
        _, pred = torch.max(final, 1)

        pred = pred.cpu().data.numpy()[0]
        # After we apply the pallete, the image is in RGB format
        image_np = self._pallete[pred.squeeze()]

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, msg.timestamp,
                                                     runtime))
        frame = SegmentedFrame(image_np, 'cityscapes', msg.frame.camera_setup)
        if self._flags.visualize_segmentation_output:
            frame.visualize(self._name, msg.timestamp)
        segmented_stream.send(
            SegmentedFrameMessage(msg.timestamp, frame, runtime))

"""Implements an operator that semantically segments frames."""
import time

import drn.segment
from drn.segment import DRNSeg

import erdos

from pylot.perception.messages import SegmentedFrameMessage
from pylot.perception.segmentation.segmented_frame import SegmentedFrame

import torch
from torch.autograd import Variable


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
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_stream, segmented_stream, flags):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [segmented_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
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

    @erdos.profile_method()
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
            msg.timestamp, self.config.name))
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
        frame = SegmentedFrame(image_np, 'cityscapes', msg.frame.camera_setup)
        segmented_stream.send(
            SegmentedFrameMessage(msg.timestamp, frame, runtime))

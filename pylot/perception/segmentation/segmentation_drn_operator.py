"""Implements an operator that semantically segments frames."""
import time

import drn.segment
from drn.segment import DRNSeg

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext

from pylot.perception.messages import SegmentedMessageTuple
from pylot.perception.segmentation.segmented_frame import SegmentedFrame

import torch
from torch.autograd import Variable


class SegmentationDRNOperator(OneInOneOut):
    """Semantically segments frames using a DRN segmentation model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, flags):
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

    def on_data(self, context: OneInOneOutContext, data: SegmentedFrame):
        """Invoked upon the receipt of a message on the camera stream."""
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        start_time = time.time()
        assert data.encoding == 'BGR', 'Expects BGR frames'
        image = torch.from_numpy(data.frame.transpose(
            [2, 0, 1])).unsqueeze(0).float()

        with torch.no_grad():
            image_var = Variable(image, requires_grad=False)

            final = self._model(image_var)[0]
            _, pred = torch.max(final, 1)

            pred = pred.cpu().data.numpy()[0]
            # After we apply the pallete, the image is in RGB format
            image_np = self._pallete[pred.squeeze()]

            # Get runtime in ms.
            runtime = (time.time() - start_time) * 1000
            frame = SegmentedFrame(image_np, 'cityscapes', data.camera_setup)
            context.write_stream.send(
                erdos.Message(context.timestamp,
                              SegmentedMessageTuple(frame, runtime)))

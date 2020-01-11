from absl import flags
import drn.segment
from drn.segment import DRNSeg
import erdos
from torch.autograd import Variable
import time
import torch

from pylot.perception.messages import SegmentedFrameMessage
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.utils import time_epoch_ms

flags.DEFINE_string('segmentation_model_path',
                    'dependencies/models/drn_d_22_cityscapes.pth',
                    'Path to the model')
flags.DEFINE_float('segmentation_drn_gpu_memory_fraction', 0.2,
                   'GPU memory fraction allocated to DRN segmentation')


class SegmentationDRNOperator(erdos.Operator):
    """ Subscribes to a camera stream, and segments frames using DRN."""
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
        self._pallete = drn.segment.CARLA_PALETTE
        # TODO(ionel): Figure out how to set GPU memory fraction.
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
        segmented_stream = erdos.WriteStream()
        return [segmented_stream]

    def on_msg_camera_stream(self, msg, segmented_stream):
        """Camera stream callback method.
        Invoked upon the receipt of a message on the camera stream.
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
        frame = SegmentedFrame(image_np, 'carla')
        if self._flags.visualize_segmentation_output:
            frame.visualize(self._name, msg.timestamp)
        segmented_stream.send(
            SegmentedFrameMessage(frame, msg.timestamp, runtime))

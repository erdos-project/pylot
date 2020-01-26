"""Implements an operator that estimates depth using two cameras."""

from absl import flags
import erdos
import cv2
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn

import anynet.anynet
from anynet import preprocess

from pylot.perception.depth_frame import DepthFrame
from pylot.perception.messages import DepthFrameMessage
from pylot.simulation.sensor_setup import CameraSetup
from pylot.utils import time_epoch_ms

flags.DEFINE_string('depth_estimation_model_path', 'dependencies/anynet/',
                    'Path to AnyNet depth estimation model')


class DepthEstimationOperator(erdos.Operator):
    """Estimates depth using two cameras, and AnyNet neural network.

    Args:
        left_camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            left camera frames are received.
        right_camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            right camera frames are received.
        depth_camera_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends computed depth frames.
        name (:obj:`str`): The name of the operator.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the center
            camera relative to the ego-vehicle.
        fov(:obj:`int`): Field of view of the center camera.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 left_camera_stream,
                 right_camera_stream,
                 depth_estimation_stream,
                 name,
                 transform,
                 fov,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):

        left_camera_stream.add_callback(self.on_left_camera_msg)
        right_camera_stream.add_callback(self.on_right_camera_msg)
        erdos.add_watermark_callback([left_camera_stream, right_camera_stream],
                                     [depth_estimation_stream],
                                     self.compute_depth)
        self._name = name
        self._flags = flags
        self._left_imgs = {}
        self._right_imgs = {}
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._transform = transform
        self._fov = fov
        # Load AnyNet
        model = anynet.anynet.AnyNet()
        model = nn.DataParallel(model).cuda()
        pretrained = os.path.join(self._flags.depth_estimation_model_path,
                                  'results/pretrained_anynet/checkpoint.tar')
        resume = os.path.join(self._flags.depth_estimation_model_path,
                              'results/finetune_anynet/checkpoint.tar')
        if os.path.isfile(pretrained):
            checkpoint = torch.load(pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            self._logger.warning('No pretrained Anynet model')

        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            self._logger.warning('No Anynet checkpoint available')

        self._model = model

    @staticmethod
    def connect(left_camera_stream, right_camera_stream):
        """Connects the operator to other streams.

        Args:
            left_camera_stream (:py:class:`erdos.ReadStream`): The stream on
                which left camera frames are received.
            right_camera_stream (:py:class:`erdos.ReadStream`): The stream on
                which right camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which depth frames are
            sent.
        """
        depth_estimation_stream = erdos.WriteStream()
        return [depth_estimation_stream]

    def on_left_camera_msg(self, msg):
        self._logger.debug('@{}: {} received left camera message'.format(
            msg.timestamp, self._name))
        img = msg.frame.as_rgb_numpy_array()
        img = preprocess.crop(img)
        processed = preprocess.get_transform(augment=False)
        img = processed(img)
        self._left_imgs[msg.timestamp] = img

    def on_right_camera_msg(self, msg):
        self._logger.debug('@{}: {} received right camera message'.format(
            msg.timestamp, self._name))
        img = msg.frame.as_rgb_numpy_array()
        img = preprocess.crop(img)
        processed = preprocess.get_transform(augment=False)
        img = processed(img)
        self._right_imgs[msg.timestamp] = img

    def compute_depth(self, timestamp, depth_estimation_stream):
        self._logger.debug('@{}: {} received watermark'.format(
            timestamp, self._name))
        start_time = time.time()

        imgL = self._left_imgs.pop(timestamp)
        imgR = self._right_imgs.pop(timestamp)

        cudnn.benchmark = False
        self._model.eval()
        imgL = imgL.float().cuda().unsqueeze(0)
        imgR = imgR.float().cuda().unsqueeze(0)
        with torch.no_grad():
            outputs = self._model(imgL, imgR)
            output = torch.squeeze(outputs[2], 1)
        output = output.squeeze().cpu().numpy()
        # Process the output (disparity) to depth, model-dependent
        depth = preprocess.disp2depth(output)
        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, timestamp,
                                                     runtime))

        if self._flags.visualize_depth_est:
            cv2.imshow(self._name, output)
            cv2.waitKey(1)

        camera_setup = CameraSetup("depth_estimation",
                                   "estimation.anynet",
                                   depth.shape[1],
                                   depth.shape[0],
                                   self._transform,
                                   fov=self._fov)
        depth_estimation_stream.send(
            DepthFrameMessage(timestamp, DepthFrame(depth, camera_setup)))

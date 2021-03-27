"""Implements an operator that estimates depth using two cameras."""
import os

import AnyNet.models.anynet as anynet
from AnyNet.utils import preprocess

from PIL import Image

import cv2

import erdos

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class AnyNetArgs(object):
    def __init__(self):
        self.init_channels = 1
        self.maxdisplist = [12, 3, 3]
        self.spn_init_channels = 1
        self.nblocks = 2
        self.layers_3d = 4
        self.channels_3d = 4
        self.growth_rate = [4, 1, 1]
        self.with_spn = False


class DepthEstimationOperator(erdos.Operator):
    """Estimates depth using two cameras, and AnyNet neural network.

    Args:
        left_camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            left camera frames are received.
        right_camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            right camera frames are received.
        depth_camera_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends computed depth frames.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the center
            camera relative to the ego-vehicle.
        fov(:obj:`int`): Field of view of the center camera.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, left_camera_stream, right_camera_stream,
                 depth_estimation_stream, transform, fov, flags):
        left_camera_stream.add_callback(self.on_left_camera_msg)
        right_camera_stream.add_callback(self.on_right_camera_msg)
        erdos.add_watermark_callback([left_camera_stream, right_camera_stream],
                                     [depth_estimation_stream],
                                     self.compute_depth)
        self._flags = flags
        self._left_imgs = {}
        self._right_imgs = {}
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._transform = transform
        self._fov = fov
        # Load AnyNet
        model = anynet.AnyNet(AnyNetArgs())
        model = nn.DataParallel(model).cuda()
        pretrained = os.path.join(self._flags.depth_estimation_model_path,
                                  'checkpoint/sceneflow/sceneflow.tar')
        resume = os.path.join(self._flags.depth_estimation_model_path,
                              'checkpoint/kitti2015_ck/checkpoint.tar')
        if os.path.isfile(pretrained):
            torch.load(pretrained)
        else:
            self._logger.warning('No pretrained Anynet model')

        if os.path.isfile(resume):
            torch.load(resume)
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

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_left_camera_msg(self, msg):
        self._logger.debug('@{}: {} received left camera message'.format(
            msg.timestamp, self.config.name))
        img = Image.fromarray(msg.frame.as_rgb_numpy_array().astype('uint8'),
                              'RGB')
        w, h = img.size
        img = img.crop((w - 960, h - 544, w, h))
        #        img = preprocess.scale_crop(img)
        processed = preprocess.get_transform(augment=False)
        img = processed(img)
        self._left_imgs[msg.timestamp] = img

    def on_right_camera_msg(self, msg):
        self._logger.debug('@{}: {} received right camera message'.format(
            msg.timestamp, self.config.name))
        img = Image.fromarray(msg.frame.as_rgb_numpy_array().astype('uint8'),
                              'RGB')
        #        img = preprocess.scale_crop(img)
        w, h = img.size
        img = img.crop((w - 960, h - 544, w, h))
        processed = preprocess.get_transform(augment=False)
        img = processed(img)
        self._right_imgs[msg.timestamp] = img

    @erdos.profile_method()
    def compute_depth(self, timestamp, depth_estimation_stream):
        self._logger.debug('@{}: {} received watermark'.format(
            timestamp, self.config.name))
        if timestamp.is_top:
            return
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
        # depth = preprocess.disp2depth(output)

        cv2.imshow(self.config.name, output)
        cv2.waitKey(1)

        # camera_setup = CameraSetup("depth_estimation",
        #                            "estimation.anynet",
        #                            depth.shape[1],
        #                            depth.shape[0],
        #                            self._transform,
        #                            fov=self._fov)
        # depth_estimation_stream.send(
        #     DepthFrameMessage(timestamp, DepthFrame(depth, camera_setup)))

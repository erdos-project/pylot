"""Implements an operator that estimates depth using two cameras."""
import os

import AnyNet.models.anynet as anynet
from AnyNet.utils import preprocess

from PIL import Image

import cv2

import erdos
from erdos.context import TwoInOneOutContext
from erdos.operator import TwoInOneOut

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


class DepthEstimationOperator(TwoInOneOut):
    """Estimates depth using two cameras, and AnyNet neural network.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the center
            camera relative to the ego-vehicle.
        fov(:obj:`int`): Field of view of the center camera.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, transform, fov, flags):
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

    def on_left_data(self, context: TwoInOneOutContext, data: erdos.Message):
        self._logger.debug('@{}: {} received left camera message'.format(
            context.timestamp, self.config.name))
        img = Image.fromarray(data.as_rgb_numpy_array().astype('uint8'), 'RGB')
        w, h = img.size
        img = img.crop((w - 960, h - 544, w, h))
        #        img = preprocess.scale_crop(img)
        processed = preprocess.get_transform(augment=False)
        img = processed(img)
        self._left_imgs[context.timestamp] = img

    def on_right_data(self, context: TwoInOneOutContext, data: erdos.Message):
        self._logger.debug('@{}: {} received right camera message'.format(
            context.timestamp, self.config.name))
        img = Image.fromarray(data.as_rgb_numpy_array().astype('uint8'), 'RGB')
        #        img = preprocess.scale_crop(img)
        w, h = img.size
        img = img.crop((w - 960, h - 544, w, h))
        processed = preprocess.get_transform(augment=False)
        img = processed(img)
        self._right_imgs[context.timestamp] = img

    def on_watermark(self, context: TwoInOneOutContext):
        self._logger.debug('@{}: {} received watermark'.format(
            context.timestamp, self.config.name))
        if context.timestamp.is_top:
            return
        imgL = self._left_imgs.pop(context.timestamp)
        imgR = self._right_imgs.pop(context.timestamp)
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
        # context.write_stream.send(
        #     DepthFrameMessage(timestamp, DepthFrame(depth, camera_setup)))

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

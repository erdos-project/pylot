from collections import deque
import cv2
from torch.autograd import Variable
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import threading
import matplotlib.pyplot as plt

import anynet.anynet
from anynet import preprocess

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

import pylot.simulation.utils
from pylot.simulation.messages import DepthFrameMessage
from pylot.utils import add_timestamp, create_depth_est_stream, is_camera_stream, rgb_to_bgr, bgra_to_bgr


class DepthEstOperator(Op):
    """ Subscribes to mid and right camera streams, and estimates depth using AnyNet."""
    def __init__(self,
                 name,
                 output_stream_name,
                 transform,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(DepthEstOperator, self).__init__(name)
        self._flags = flags
        self._left_imgs = deque()
        self._right_imgs = deque()
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name
        self._transform = transform
        self._lock = threading.Lock()
        #load AnyNet
        model = anynet.anynet.AnyNet()
        model = nn.DataParallel(model).cuda()
        
        path_to_anynet = self._flags.depth_est_model_path
        pretrained = path_to_anynet + 'results/pretrained_anynet/checkpoint.tar'
        resume = path_to_anynet + 'results/finetune_anynet/checkpoint.tar'
        
        if os.path.isfile(pretrained):
            checkpoint = torch.load(pretrained)
            model.load_state_dict(checkpoint['state_dict'])

        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            
        self._model = model

    @staticmethod
    def setup_streams(input_streams, output_stream_name,
                      left_camera_name, right_camera_name):
        # Select camera input streams.
        camera_streams = input_streams.filter(is_camera_stream)
        camera_streams.filter_name(left_camera_name).add_callback(
            DepthEstOperator.on_left_camera_msg)
        camera_streams.filter_name(right_camera_name).add_callback(
            DepthEstOperator.on_right_camera_msg)
        return [create_depth_est_stream(output_stream_name)]

    def on_left_camera_msg(self, msg):
        with self._lock:
            img = rgb_to_bgr(msg.frame).astype(np.uint8) # used as bgr_to_rgb, need RGB frames
            img = img[200:500] # crop
            processed = preprocess.get_transform(augment=False)  
            img = processed(img)
            self._left_imgs.append(img)
            if self._right_imgs:
                self.eval_depth(msg)
        
    def on_right_camera_msg(self, msg):
        with self._lock:
            img = rgb_to_bgr(msg.frame).astype(np.uint8) # used as bgr_to_rgb, need RGB frames
            img = img[200:500] # crop
            processed = preprocess.get_transform(augment=False)  
            img = processed(img)
            self._right_imgs.append(img)
            if self._left_imgs:
                self.eval_depth(msg)
            
    def eval_depth(self, msg):
        start_time = time.time()

        imgL = self._left_imgs.popleft()
        imgR = self._right_imgs.popleft()

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
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))

        if self._flags.visualize_depth_est:
            plt.imshow(output, cmap = 'viridis')
            plt.show()
    
        output_msg = DepthFrameMessage(depth, self._transform, 90, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)
        
    def execute(self):
        """Operator execute entry method."""
        # Ensures that the operator runs continuously.
        self.spin()

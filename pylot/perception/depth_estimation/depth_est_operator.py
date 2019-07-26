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

from pylot.perception.messages import DepthFrameMessage
from pylot.utils import add_timestamp, create_depth_est_stream, is_camera_stream, rgb_to_bgr, bgra_to_bgr

args = {'init_channels':1,
        'maxdisplist':[12, 3, 3],
        'spn_init_channels':8,
        'nblocks':1,
        'layers_3d':4,
        'channels_3d':4,
        'growth_rate':[4,1,1],
        'with_spn':0,
        'resume':'/home/carla/code/pylot/dependencies/anynet/results/finetune_anynet/checkpoint.tar',
        'pretrained':'/home/carla/code/pylot/dependencies/anynet/results/pretrained_anynet/checkpoint.tar',
        'lr':1e-3
       }

class DepthEstOperator(Op):
    """ Subscribes to mid and right camera streams, and estimates depth using AnyNet."""
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(DepthEstOperator, self).__init__(name)
        self._flags = flags
        self._left_camera_msgs = deque()
        self._right_camera_msgs = deque()
        self._left_imgs = deque()
        self._right_imgs = deque()
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name
        self._lock = threading.Lock()
        #load AnyNet
        model = anynet.anynet.AnyNet(args)
        model = nn.DataParallel(model).cuda()

        if os.path.isfile(args['pretrained']):
            checkpoint = torch.load(args['pretrained'])
            model.load_state_dict(checkpoint['state_dict'])

        if os.path.isfile(args['resume']):
            checkpoint = torch.load(args['resume'])
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
        input_streams.add_completion_callback(
            DepthEstOperator.on_watermark)
        return [create_depth_est_stream(output_stream_name)]

    def on_left_camera_msg(self, msg):
        with self._lock:
            self._left_camera_msgs.append(msg)
            img = rgb_to_bgr(msg.frame).astype(np.uint8) # used as bgr_to_rgb, need RGB frames
            img = img[200:500] # crop
            processed = preprocess.get_transform(augment=False)  
            img = processed(img)
            self._left_imgs.append(img)
            if self._right_imgs:
                self.eval_depth(msg)
        
    def on_right_camera_msg(self, msg):
        with self._lock:
            self._right_camera_msgs.append(msg)
            img = rgb_to_bgr(msg.frame).astype(np.uint8) # used as bgr_to_rgb, need RGB frames
            img = img[200:500] # crop
            processed = preprocess.get_transform(augment=False)  
            img = processed(img)
            self._right_imgs.append(img)
            if self._left_imgs:
                self.eval_depth(msg)
        
    def on_watermark(self, msg):
        with self._lock:
            left_camera_msg = self._left_camera_msgs.popleft()
            right_camera_msg = self._right_camera_msgs.popleft()

            assert (left_camera_msg.timestamp == right_camera_msg.timestamp)
            
    def eval_depth(self, msg):
        start_time = time.time()

        imgL = self._left_imgs.popleft()
        imgR = self._right_imgs.popleft()

        model = self._model
        cudnn.benchmark = False    
        model.eval()
        imgL = imgL.float().cuda().unsqueeze(0)
        imgR = imgR.float().cuda().unsqueeze(0)
        with torch.no_grad():
            outputs = model(imgL, imgR)
            output = torch.squeeze(outputs[2], 1)
        output = output.squeeze().cpu().numpy()
        depth = 1000*np.reciprocal(output)
        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))

        if self._flags.visualize_depth_est:
            print(output)
            plt.imshow(output, cmap = 'viridis')
            plt.show()
            cv2.waitKey(1)

        output_msg = DepthFrameMessage(depth, runtime, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)
        
    def execute(self):
        """Operator execute entry method."""
        # Ensures that the operator runs continuously.
        self.spin()

import logging
import math
import os
import sys
sys.path.append("{}/dependencies/lanenet-lane-detection".format(
    os.getenv("PYLOT_HOME")))

import cv2

import erdos

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

import numpy as np

import pylot.utils
from pylot.perception.detection.lane import Lane

import tensorflow as tf


class LanenetDetectionOperator(erdos.Operator):
    def __init__(self, camera_stream, detected_lanes_stream, flags):
        camera_stream.add_callback(self.on_camera_frame,
                                   [detected_lanes_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        pylot.utils.set_tf_loglevel(logging.ERROR)
        self._input_tensor = tf.placeholder(dtype=tf.float32,
                                            shape=[1, 256, 512, 3],
                                            name='input_tensor')
        net = lanenet.LaneNet(phase='test')
        self._binary_seg_ret, self._instance_seg_ret = net.inference(
            input_tensor=self._input_tensor, name='LaneNet')
        self._gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=flags.
            lane_detection_gpu_memory_fraction,
            allow_growth=True,
            allocator_type='BFC')
        self._tf_session = tf.Session(config=tf.ConfigProto(
            gpu_options=self._gpu_options))
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(0.9995)
            variables_to_restore = variable_averages.variables_to_restore()

        self._postprocessor = lanenet_postprocess.LaneNetPostProcessor()
        saver = tf.train.Saver(variables_to_restore)
        with self._tf_session.as_default():
            saver.restore(sess=self._tf_session,
                          save_path=flags.lanenet_detection_model_path)

    @staticmethod
    def connect(camera_stream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.LanesMessage` messages.
        """
        detected_lanes_stream = erdos.WriteStream()
        return [detected_lanes_stream]

    @erdos.profile_method()
    def on_camera_frame(self, msg, detected_lanes_stream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg: A :py:class:`~pylot.perception.messages.FrameMessage`.
            detected_lanes_stream (:py:class:`erdos.WriteStream`): Stream on
                which the operator sends
                :py:class:`~pylot.perception.messages.LanesMessage` messages.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        image = cv2.resize(msg.frame.as_rgb_numpy_array(), (512, 256),
                           interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        binary_seg_image, instance_seg_image = self._tf_session.run(
            [self._binary_seg_ret, self._instance_seg_ret],
            feed_dict={self._input_tensor: [image]})

        postprocess_result = self._postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=msg.frame.frame)
        # mask_image = postprocess_result['mask_image']
        # for i in range(4):
        #     instance_seg_image[0][:, :, i] = \
        #         minmax_scale(instance_seg_image[0][:, :, i])
        # embedding_image = np.array(instance_seg_image[0], np.uint8)

        lanes = postprocess_result['lanes']
        ego_lane_markings = []
        for lane in lanes:
            ego_markings = self.lane_to_ego_coordinates(
                lane, msg.frame.camera_setup)
            ego_lane_markings.append([
                pylot.utils.Transform(loc, pylot.utils.Rotation())
                for loc in ego_markings
            ])
        # Sort the lane markings from left to right.
        ego_lane_markings = sorted(ego_lane_markings,
                                   key=lambda lane: lane[-1].location.y)
        detected_lanes = []
        for index, lane in enumerate(ego_lane_markings):
            if index > 0:
                lane = Lane(ego_lane_markings[index - 1], lane)
                detected_lanes.append(lane)
        self._logger.debug('@{}: Detected {} lanes'.format(
            msg.timestamp, len(detected_lanes)))
        detected_lanes_stream.send(erdos.Message(msg.timestamp,
                                                 detected_lanes))
        detected_lanes_stream.send(erdos.WatermarkMessage(msg.timestamp))

        # plt.figure('binary_image')
        # plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        # plt.figure('instance_image')
        # plt.imshow(embedding_image[:, :, (2, 1, 0)])
        # plt.figure('mask_image')
        # plt.imshow(msg.frame.frame[:, :, (2, 1, 0)])
        # plt.show()

    def lane_to_ego_coordinates(self, lane, camera_setup):
        inverse_intrinsic_matrix = np.linalg.inv(
            camera_setup.get_intrinsic_matrix())
        camera_ground_height = camera_setup.get_transform().location.z
        pitch = -math.radians(camera_setup.get_transform().rotation.pitch)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        pitch_matrix = np.array([[1, 0, 0], [0, cos_pitch, sin_pitch],
                                 [0, -sin_pitch, cos_pitch]])
        ego_lane = []
        locations = []
        for x, y in lane:
            # Project the 2D pixel location into 3D space, onto the z=1 plane.
            p3d = np.dot(inverse_intrinsic_matrix, np.array([[x], [y], [1.0]]))
            rotate_point = np.dot(pitch_matrix, p3d)
            scale = camera_ground_height / rotate_point[1][0]
            loc = pylot.utils.Location(rotate_point[0][0] * scale,
                                       rotate_point[1][0] * scale,
                                       rotate_point[2][0] * scale)
            locations.append(loc)
        to_world_transform = camera_setup.get_unreal_transform()
        ego_lane = to_world_transform.transform_locations(locations)
        # Reset z = ground.
        for loc in ego_lane:
            loc.z = 0
        return ego_lane


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

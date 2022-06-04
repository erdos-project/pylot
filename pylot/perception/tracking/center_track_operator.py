import cv2

import erdos

import numpy as np

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    BoundingBox3D, OBSTACLE_LABELS
from pylot.perception.messages import ObstaclesMessage

import torch


class CenterTrackOperator(erdos.Operator):
    def __init__(self, camera_stream, obstacle_tracking_stream, flags,
                 camera_setup):
        from dataset.dataset_factory import get_dataset
        from model.model import create_model, load_model
        from opts import opts
        from utils.tracker import Tracker

        camera_stream.add_callback(self.on_frame_msg,
                                   [obstacle_tracking_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._camera_setup = camera_setup
        # TODO(ionel): Might have to filter labels when running with a coco
        # and a nuscenes model.
        num_classes = {
            'kitti_tracking': 3,
            'coco': 90,
            'mot': 1,
            'nuscenes': 10
        }
        # Other flags:
        # 1) --K ; max number of output objects.
        # 2) --fix_short ; resizes the height of the image to fix short, and
        # the width such the aspect ratio is maintained.
        # 3) --pre_hm ; pre heat map.
        # 4) --input_w; str(camera_setup.width)
        # 5) --input_h; str(camera_setup.height)
        args = [
            'tracking', '--load_model', flags.center_track_model_path,
            '--dataset', flags.center_track_model, '--test_focal_length',
            str(int(camera_setup.get_focal_length())), '--out_thresh',
            str(flags.obstacle_detection_min_score_threshold), '--pre_thresh',
            str(flags.obstacle_detection_min_score_threshold), '--new_thresh',
            str(flags.obstacle_detection_min_score_threshold),
            '--track_thresh',
            str(flags.obstacle_detection_min_score_threshold), '--max_age',
            str(flags.obstacle_track_max_age), '--num_classes',
            str(num_classes[flags.center_track_model]), '--tracking',
            '--hungarian'
        ]
        opt = opts().init(args)
        gpu = True
        if gpu:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        self.opt = opt
        self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
        self.model = load_model(self.model, opt.load_model, opt)
        self.model = self.model.to(self.opt.device)
        self.model.eval()

        self.trained_dataset = get_dataset(opt.dataset)
        self.mean = np.array(self.trained_dataset.mean,
                             dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.trained_dataset.std,
                            dtype=np.float32).reshape(1, 1, 3)
        self.rest_focal_length = self.trained_dataset.rest_focal_length \
            if self.opt.test_focal_length < 0 else self.opt.test_focal_length
        self.flip_idx = self.trained_dataset.flip_idx
        self.cnt = 0
        self.pre_images = None
        self.pre_image_ori = None
        self.tracker = Tracker(opt)

    @staticmethod
    def connect(camera_stream):
        obstacle_tracking_stream = erdos.WriteStream()
        return [obstacle_tracking_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    @erdos.profile_method()
    def on_frame_msg(self, msg, obstacle_tracking_stream):
        """Invoked when a FrameMessage is received on the camera stream."""
        self._logger.debug('@{}: {} received frame'.format(
            msg.timestamp, self.config.name))
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        image_np = msg.frame.as_bgr_numpy_array()
        results = self.run_model(image_np)
        obstacles = []
        for res in results:
            track_id = res['tracking_id']
            bbox = res['bbox']
            score = res['score']
            (label_id, ) = res['class'] - 1,
            if label_id > 80:
                continue
            label = self.trained_dataset.class_name[label_id]
            if label in ['Pedestrian', 'pedestrian']:
                label = 'person'
            elif label == 'Car':
                label = 'car'
            elif label == 'Cyclist':
                label == 'bicycle'
            if label in OBSTACLE_LABELS:
                bounding_box_2D = BoundingBox2D(bbox[0], bbox[2], bbox[1],
                                                bbox[3])
                bounding_box_3D = None
                if 'dim' in res and 'loc' in res and 'rot_y' in res:
                    bounding_box_3D = BoundingBox3D.from_dimensions(
                        res['dim'], res['loc'], res['rot_y'])
                obstacles.append(
                    Obstacle(bounding_box_3D,
                             score,
                             label,
                             track_id,
                             bounding_box_2D=bounding_box_2D))
        obstacle_tracking_stream.send(
            ObstaclesMessage(msg.timestamp, obstacles, 0))

    def run_model(self, image_np, meta={}):
        images, meta = self.pre_process(image_np, meta)
        images = images.to(self.opt.device,
                           non_blocking=self.opt.non_block_test)
        pre_hms, pre_inds = None, None
        if self.pre_images is None:
            self.pre_images = images
            self.tracker.init_track(meta['pre_dets'] if 'pre_dets' in
                                    meta else [])
        if self.opt.pre_hm:
            pre_hms, pre_inds = self._get_additional_inputs(
                self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)

        output, dets = self.process(images, self.pre_images, pre_hms, pre_inds)
        detections = self.post_process(dets, meta)

        # Filter out detections below threshold.
        detections = [
            det for det in detections if det['score'] > self.opt.out_thresh
        ]
        torch.cuda.synchronize()
        public_det = meta['cur_dets'] if self.opt.public_det else None
        # Add tracking id to results.
        results = self.tracker.step(detections, public_det)
        self.pre_images = images
        return results

    def process(self, images, pre_images=None, pre_hms=None, pre_inds=None):
        from model.decode import generic_decode
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images, pre_images, pre_hms)[-1]
            output = self._sigmoid_output(output)
            output.update({'pre_inds': pre_inds})
            if self.opt.flip_test:
                output = self._flip_output(output)
            torch.cuda.synchronize()
            dets = generic_decode(output, K=self.opt.K, opt=self.opt)
            torch.cuda.synchronize()
            for k in dets:
                dets[k] = dets[k].detach().cpu().numpy()
        return output, dets

    def pre_process(self, image, input_meta={}):
        """
        Crop, resize, and normalize image. Gather meta data for post
        processing and tracking.
        """
        from utils.image import get_affine_transform
        resized_image, c, s, inp_width, inp_height, height, width = \
            self._transform_scale(image)
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        out_height = inp_height // self.opt.down_ratio
        out_width = inp_width // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        inp_image = cv2.warpAffine(resized_image,
                                   trans_input, (inp_width, inp_height),
                                   flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(
            np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height,
                                                      inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {
            'calib': np.array(input_meta['calib'], dtype=np.float32) if 'calib'
            in input_meta else self._get_default_calib(width, height)
        }
        meta.update({
            'c': c,
            's': s,
            'height': height,
            'width': width,
            'out_height': out_height,
            'out_width': out_width,
            'inp_height': inp_height,
            'inp_width': inp_width,
            'trans_input': trans_input,
            'trans_output': trans_output
        })
        if 'pre_dets' in input_meta:
            meta['pre_dets'] = input_meta['pre_dets']
        if 'cur_dets' in input_meta:
            meta['cur_dets'] = input_meta['cur_dets']
        return images, meta

    def _get_default_calib(self, width, height):
        calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                          [0, self.rest_focal_length, height / 2, 0],
                          [0, 0, 1, 0]])
        return calib

    def _transform_scale(self, image, scale=1):
        """
        Prepare input image in different testing modes.
        Currently support: fix short size/ center crop to a fixed size/
        keep original resolution but pad to a multiplication of 32.
        """
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_short > 0:
            if height < width:
                inp_height = self.opt.fix_short
                inp_width = (int(width / height * self.opt.fix_short) +
                             63) // 64 * 64
            else:
                inp_height = (int(height / width * self.opt.fix_short) +
                              63) // 64 * 64
                inp_width = self.opt.fix_short
            c = np.array([width / 2, height / 2], dtype=np.float32)
            s = np.array([width, height], dtype=np.float32)
        elif self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
            # s = np.array([inp_width, inp_height], dtype=np.float32)
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, c, s, inp_width, inp_height, height, width

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = output['hm'].sigmoid_()
        if 'hm_hp' in output:
            output['hm_hp'] = output['hm_hp'].sigmoid_()
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            output['dep'] *= self.opt.depth_scale
        return output

    def post_process(self, dets, meta):
        from utils.post_process import generic_post_process
        dets = generic_post_process(self.opt, dets, [meta['c']], [meta['s']],
                                    meta['out_height'], meta['out_width'],
                                    self.opt.num_classes, [meta['calib']],
                                    meta['height'], meta['width'])
        self.this_calib = meta['calib']
        return dets[0]

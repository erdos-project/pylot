"""Implements an operator that eveluates detection output."""
import json
from datetime import datetime

import erdos

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pylot.perception.base_perception_eval_operator import \
    BasePerceptionEvalOperator, ScoringModule
from pylot.utils import verify_keys_in_dict


class DetectionEvalOperator(BasePerceptionEvalOperator):
    """Operator that computes accuracy metrics using detected obstacles.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            detected obstacles are received.
        ground_obstacles_stream: The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received from the simulator.
        finished_indicator_stream: Stream on which the operator notifies
            when it finished execution.
        evaluate_timely (:obj:`bool`): Compute timely or non-timely accuracy.
        frame_gap (:obj:`int`): Time in ms between frames.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, prediction_stream: erdos.ReadStream,
                 ground_truth_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream,
                 evaluate_timely: bool, matching_policy: str, frame_gap: int,
                 flags):
        super().__init__(prediction_stream, ground_truth_stream,
                         finished_indicator_stream, evaluate_timely,
                         matching_policy, frame_gap,
                         CocoDetectionScoringModule, flags)


class CocoDetectionScoringModule(ScoringModule):
    def __init__(self, flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(
            "coco_detection_scoring_module", self._flags.log_file_name)
        label_list = ["person", "bicycle", "car", "motorcycle"]
        self.coco_eval = OnlineCOCOEval(label_list)
        self.get_summary_counter = 0

    def get_scores(self):
        """
        Every FLAGS.detection_eval_freq calls to this function (starting
        with the first call) it returns eval aggregate of the last
        FLAGS.detection_eval_lookback images as a dictionary.
        """
        self.get_summary_counter += 1
        if (self.get_summary_counter -
                1) % self._flags.detection_eval_freq != 0:
            # check counter - 1 so first call goes through
            return {}
        if len(self.coco_eval.images) < self._flags.detection_eval_lookback:
            # Skipping aggregate map watermark; there are num_images < n images
            return {}
        result_dict = self.coco_eval.evaluate_last_n(
            self._flags.detection_eval_lookback)
        return {"coco_" + k: v for k, v in result_dict.items()}

    def add_datapoint(self, obstacles, ground_obstacles):
        """
        Adds datapoint to internal datastructure for bookkeping images and
        their lables/predictions.
        """
        obstacles = self.__filter_obstacles(obstacles)
        ground_obstacles = self.__filter_obstacles(ground_obstacles)

        def obstacle_to_dict(obstacle):
            mn = obstacle.bounding_box_2D.get_min_point()
            r = {
                "category_id": obstacle.label,
                "bbox": [
                    mn.x, mn.y,
                    obstacle.bounding_box_2D.get_height(),
                    obstacle.bounding_box_2D.get_width()
                ]
            }
            r["score"] = obstacle.confidence
            return r

        lables_dict_list = \
            [obstacle_to_dict(go)
             for go in ground_obstacles if go.label in self.from_dataset]
        pred_dict_list = \
            [obstacle_to_dict(o)
             for o in obstacles if o.label in self.from_model]
        self.coco_eval.add_image_label_prediction(
            image_dict={
                "width": self._flags.camera_image_width,
                "height": self._flags.camera_image_height,
                "file_name": None
            },
            lables_dict_list=lables_dict_list,
            pred_dict_list=pred_dict_list)

    def __filter_obstacles(self, obstacles):
        vehicles, people, _ = self.__get_obstacles_by_category(obstacles)
        return vehicles + people

    def __get_obstacles_by_category(self, obstacles):
        """Divides perception.detection.obstacle.Obstacle by labels."""
        vehicles = []
        people = []
        traffic_lights = []
        for obstacle in obstacles:
            if obstacle.is_vehicle():
                vehicles.append(obstacle)
            elif obstacle.is_person():
                people.append(obstacle)
            elif obstacle.is_traffic_light():
                traffic_lights.append(obstacle)
            else:
                self._logger.warning('Unexpected label {}'.format(
                    obstacle.label))
        return vehicles, people, traffic_lights


# groundtruth_dataset_template = {
#     "info": {
#         "version": "0.0.1",
#         "description": "AD_evaluation",
#         "date_created": datetime.now(),
#     },
#     "images": [
#         {
#             "id": int, "width": int, "height": int, "file_name": str,
#         }
#     ],
#     "annotations": [
#         {
#             "id": int, "image_id": int, "category_id": int, "area": float,
#             "bbox": [x, y, width, height], "iscrowd": 0
#         }
#     ],
#     "categories": [
#         {
#             "id": int, "name": str, "supercategory": str
#         }
#     ],
#     "licenses": None
# }

# predictions_template = [
#     {
#         "image_id": 42, "category_id": 18,
#         "bbox": [258.15, 41.29, 348.26, 243.78],
#         "score":0.236
#     }
# ]


class OnlineCOCOEval(object):
    def __init__(self, label_list):
        self.categories = [{
            "id": i,
            "name": l.strip(),
            "supercategory": "Pylot_object"
        } for i, l in enumerate(label_list)]
        self.model_label_map = {x["name"]: x["id"] for x in self.categories}
        self.next_image_id = 0
        self.next_annotation_id = 1
        self.images = []
        self.annotations = []
        self.preds = []

    def add_image_label_prediction(self, image_dict, lables_dict_list,
                                   pred_dict_list):
        """
        image_dict: {"width": , "height: ,"file_name":}
        lables_dict_list: [{"category_id": , "bbox": [x,y,w,h] }, ...]
        pred_dict_list: [{"category_id": , "bbox": [x,y,w,h], "score": }, ...]
        """
        img_id = self.next_image_id
        self.next_image_id += 1
        verify_keys_in_dict(["width", "height", "file_name"], image_dict)
        image_dict["id"] = img_id

        def prep_label_dict(label_dict):
            verify_keys_in_dict(["category_id", "bbox"], label_dict)
            error_msg = ("The dataset uses a label {} that doesn't show up "
                         "in the model's label map {}")
            assert label_dict["category_id"] in self.model_label_map, \
                error_msg.format(label_dict["category_id"],
                                 self.model_label_map)
            label_dict["category_id"] = self.model_label_map[
                label_dict["category_id"]]
            label_dict["image_id"] = img_id
            label_dict["area"] = label_dict["bbox"][2] * label_dict["bbox"][3]
            label_dict["iscrowd"] = 0
            label_dict["id"] = self.next_annotation_id
            self.next_annotation_id += 1
            return label_dict

        lables_dict_list = [prep_label_dict(d) for d in lables_dict_list]
        lables_dict_list = filter(lambda x: x is not None, lables_dict_list)

        def prep_pred_dict(pred_dict):
            verify_keys_in_dict(["category_id", "bbox", "score"], pred_dict)
            error_msg = ("given model prediction {} that is not in the "
                         "predefined model label map {}")
            assert pred_dict["category_id"] in self.model_label_map, \
                error_msg.format(pred_dict["category_id"],
                                 self.model_label_map)
            pred_dict["category_id"] = self.model_label_map[
                pred_dict["category_id"]]
            pred_dict["image_id"] = img_id
            return pred_dict

        pred_dict_list = [prep_pred_dict(d) for d in pred_dict_list]

        self.images.append(image_dict)
        self.annotations.extend(lables_dict_list)
        self.preds.extend(pred_dict_list)

    def evaluate_last_n(self, n: int = None, verbose=False):
        """
        if n = `None` evaluate over all images added so far
        """
        assert n is None or n > 0, "Should evaluate over at least 1 image"
        assert len(self.images) > 0, "No images to evaluate on"
        # assert len(self.annotations) > 0, "No annotations to evaluate on"
        # assert len(self.preds) > 0, "No predictions to evaluate on"
        n = -len(self.images) if n is None else -n
        images_to_use = self.images[n:]
        preds_to_use = [
            p for p in self.preds if p["image_id"] >= images_to_use[0]["id"]
        ]
        anns_to_use = [
            a for a in self.annotations
            if a["image_id"] >= images_to_use[0]["id"]
        ]
        groundtruth_dataset_template = {
            "info": {
                "version": "0.0.1",
                "description": "Pylot_evaluation",
                "date_created": datetime.now().isoformat(),
            },
            "images": images_to_use,
            "annotations": anns_to_use,
            "categories": self.categories
        }
        # labels_categories = set([a["category_id"] for a in anns_to_use])
        # pred_categories = set([a["category_id"] for a in preds_to_use])
        # preds_not_in_labels = pred_categories.difference(labels_categories)
        # assert len(preds_not_in_labels) == 0, \
        #     "The model predicts categories {} that don't show up in the dataset".format(([self.categories[idx] for idx in labels_categories],   # noqa: E501
        #     [self.categories[idx] for idx in pred_categories]))
        keys = [
            "AP_IoU=0.50:0.95_area=all_maxDets=100",
            "AP_IoU=0.50_area=all_maxDets=100",
            "AP_IoU=0.75_area=all_maxDets=100",
            "AP_IoU=0.50:0.95_area=small_maxDets=100",
            "AP_IoU=0.50:0.95_area=medium_maxDets=100",
            "AP_IoU=0.50:0.95_area=large_maxDets=100",
            "AR_IoU=0.50:0.95_area=all_maxDets=1",
            "AR_IoU=0.50:0.95_area=all_maxDets=10",
            "AR_IoU=0.50:0.95_area=all_maxDets=100",
            "AR_IoU=0.50:0.95_area=small_maxDets=100",
            "AR_IoU=0.50:0.95_area=medium_maxDets=100",
            "AR_IoU=0.50:0.95_area=large_maxDets=100",
        ]
        if verbose:
            print(
                json.dumps(groundtruth_dataset_template,
                           indent=4,
                           sort_keys=True))
            print(json.dumps(preds_to_use, indent=4, sort_keys=True))
        if len(preds_to_use) == 0 and len(anns_to_use) > 0:
            return {k: 0 for k in keys}
        cocoGt = COCO()
        cocoGt.dataset = groundtruth_dataset_template
        cocoGt.createIndex()
        cocoDt = cocoGt.loadRes(preds_to_use)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        # cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        values = cocoEval.stats
        return {k: v for k, v in zip(keys, values)}

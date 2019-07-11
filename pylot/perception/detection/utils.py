import cv2
from enum import Enum
import numpy as np
import PIL.Image as Image
try:
    import queue as queue
except ImportError:
    import Queue as queue
from skimage import measure

from pylot.utils import add_timestamp

ADJACENT_POS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

GROUND_COLOR_MAP = {
    'pedestrian': [0, 128, 0],
    'vehicle': [128, 0, 0],
    'stop sign': [128, 128, 0],
    'speed limit': [255, 255, 0],
    'red traffic light': [0, 0, 255],
    'yellow traffic light': [0, 255, 255],
    'green traffic light': [0, 255, 0],
}

coco_bbox_color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)


class TrafficLightColor(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3
    OFF = 4


class DetectedObject(object):
    """ Class that stores info about a detected object.

    Attributes:
        corners: The corners of the bounding box.
        confidence: The confidence of the detection.
        label: The label of the detected object.
    """
    def __init__(self, corners, confidence, label):
        self.corners = corners
        self.confidence = confidence
        self.label = label

    def visualize_on_img(self, image_np, bbox_color_map, text=None):
        """ Annotate the image with the bounding box of the obstacle."""
        txt_font = cv2.FONT_HERSHEY_SIMPLEX
        (xmin, xmax, ymin, ymax) = self.corners
        if text is None:
            text = '{}{:.1f}'.format(self.label, self.confidence)
        txt_size = cv2.getTextSize(text, txt_font, 0.5, 2)[0]
        color = bbox_color_map[self.label]
        # Show bounding box.
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Show text.
        cv2.rectangle(image_np,
                      (xmin, ymin - txt_size[1] - 2),
                      (xmin + txt_size[0], ymin - 2), color, -1)
        cv2.putText(image_np, text, (xmin, ymin - 2),
                    txt_font, 0.5, (0, 0, 0), thickness=1,
                    lineType=cv2.LINE_AA)

    def get_bbox_label(self, ):
        (xmin, xmax, ymin, ymax) = self.corners
        return (self.label, ((xmin, ymin), (xmax, ymax)))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DetectedObject(label: {}, confidence: {}, bbox: {})'.format(
            self.label, self.confidence, self.corners)


class DetectedSpeedLimit(DetectedObject):
    def __init__(self, corners, limit, confidence, label):
        super(DetectedSpeedLimit, self).__init__(corners, confidence, label)
        self.limit = limit

    def visualize_on_img(self, image_np, bbox_color_map):
        text = '{} {} {:.1f}'.format(self.limit, self.label, self.confidence)
        super(DetectedSpeedLimit, self).visualize_on_img(
            image_np, bbox_color_map, text)

    def get_bbox_label(self, ):
        (xmin, xmax, ymin, ymax) = self.corners
        return (self.label + ' ' + str(self.limit),
                ((xmin, ymin), (xmax, ymax)))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DetectedSpeedLimit(label: {}, limit: {}, '\
            'confidence: {}, bbox: {})'.format(
                self.label, self.limit, self.confidence, self.corners)


def compute_miou(bboxes1, bboxes2):
    """ Compute mIoU for two lists of bounding boxes."""
    bboxes1, bboxes2 = np.array(bboxes1), np.array(bboxes2)
    x11, x12, y11, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xI1 = np.maximum(x11, np.transpose(x21))
    xI2 = np.minimum(x12, np.transpose(x22))

    yI1 = np.maximum(y11, np.transpose(y21))
    yI2 = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xI2 - xI1), 0) * np.maximum((yI2 - yI1), 0)

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

    return inter_area / (union+0.0001)


def load_coco_labels(labels_path):
    labels_map = {}
    with open(labels_path) as labels_file:
        labels = labels_file.read().splitlines()
        index = 1
        for label in labels:
            labels_map[index] = label
            index += 1
    return labels_map


def load_coco_bbox_colors(coco_labels):
    # Transform to RGB values.
    bbox_color_list = coco_bbox_color_list.reshape((-1, 3)) * 255
    # Transform to ints
    bbox_colors = [(bbox_color_list[_]).astype(np.uint8)
                   for _ in range(len(bbox_color_list))]
    bbox_colors = np.array(bbox_colors, dtype=np.uint8).reshape(
        len(bbox_colors), 1, 1, 3)

    colors = {}
    for category, label in coco_labels.items():
        colors[label] = bbox_colors[category - 1][0][0].tolist()
    return colors


def calculate_iou(ground_truth, prediction):
    """Calculate the IoU of a single predicted ground truth box."""
    x1_gt, x2_gt, y1_gt, y2_gt = ground_truth
    x1_p, x2_p, y1_p, y2_p = prediction

    if x1_p > x2_p or y1_p > y2_p:
        raise AssertionError("Prediction box is malformed? {}".format(prediction))

    if x1_gt > x2_gt or y1_gt > y2_gt:
        raise AssertionError("Ground truth box is malformed? {}".format(ground_truth))

    if x2_gt < x1_p or x2_p < x1_gt or y2_gt < y1_p or y2_p < y1_gt:
        return 0.0

    inter_x1 = max([x1_gt, x1_p])
    inter_x2 = min([x2_gt, x2_p])

    inter_y1 = max([y1_gt, y1_p])
    inter_y2 = min([y2_gt, y2_p])

    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    gt_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    pred_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    return float(inter_area) / (gt_area + pred_area - inter_area)


def get_bounding_boxes_from_segmented(frame, min_width=2, min_height=3):
    """ Extracts bounding box from frame.
    Assumes that the pixels we are interested in are set to True.
    """
    bboxes = []
    # Labels the connected segmented pixels.
    map_labeled = measure.label(frame, connectivity=1)
    # Extract the regions out of the labeled frames.
    for region in measure.regionprops(map_labeled):
        x_min = region.bbox[1]
        x_max = region.bbox[3]
        y_min = region.bbox[0]
        y_max = region.bbox[2]
        # Filter the bboxes that are extremely small.
        if x_max - x_min > min_width and y_max - y_min > min_height:
            bboxes.append((x_min, x_max, y_min, y_max))
    return bboxes


def get_prediction_results(ground_truths, predictions, iou_threshold):
    """Calculate the number of true positives, false positives and false
    negatives from the given ground truth and predictions."""
    true_pos, false_pos, false_neg = None, None, None

    # If there are no predictions, then everything is a false negative.
    if len(predictions) == 0:
        true_pos, false_pos = 0, 0
        false_neg = len(ground_truths)
        return true_pos, false_pos, false_neg

    # If there is no ground truth, everything is a false positive.
    if len(ground_truths) == 0:
        true_pos, false_neg = 0, 0
        false_pos = len(predictions)
        return true_pos, false_pos, false_neg

    # Iterate over the predictions and calculate the IOU of each prediction
    # with each ground truth.
    ious = []
    for i, prediction in enumerate(predictions):
        for j, ground_truth in enumerate(ground_truths):
            iou = calculate_iou(prediction, ground_truth)
            if iou > iou_threshold:
                ious.append((i, j, iou))

    # If no IOUs were over the threshold, return all predictions as false
    # positives and all ground truths as false negatives.
    if len(ious) == 0:
        true_pos = 0
        false_pos, false_neg = len(predictions), len(ground_truths)
    else:
        # Sort the IOUs and match each box only once.
        ground_truths_matched, predictions_matched = set(), set()
        matched = []
        for prediction, ground_truth, iou in sorted(ious, key=lambda x: x[-1], reverse=True):
            if ground_truth not in ground_truths_matched and prediction not in predictions_matched:
                ground_truths_matched.add(ground_truth)
                predictions_matched.add(prediction)
                matched.append((prediction, ground_truth, iou))

        # The matches are the true positives.
        true_pos = len(matched)
        # The unmatched predictions are the false positives.
        false_pos = len(predictions) - len(predictions_matched)
        # The umatched ground truths are the false negatives.
        false_neg = len(ground_truths) - len(ground_truths_matched)

    return true_pos, false_pos, false_neg


def get_precision_recall(true_positives, false_positives, false_negatives):
    precision, recall = None, None
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = float(true_positives) / float(true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = float(true_positives) / float(true_positives + false_negatives)

    return (precision, recall)


def get_precision_recall_at_iou(ground_truths, predictions, iou_threshold):
    true_pos, false_pos, false_neg = get_prediction_results(
        ground_truths, predictions, iou_threshold)
    return get_precision_recall(true_pos, false_pos, false_neg)


def get_pedestrian_mAP(ground_bboxes, detected_objs):
    """Return mAP with IoU threshold of 0.5"""
    # Select the pedestrians.
    confidence_bbox = []
    for detected_obj in detected_objs:
        if detected_obj.label == 'person':
            confidence_bbox.append(
                (detected_obj.confidence, detected_obj.corners))
    # Sort bboxes descending by score.
    confidence_bbox.sort()
    confidence_bbox.reverse()
    detected_bboxes = [bbox for (score, bbox) in confidence_bbox]
    # Compute recall precision. The results are sorted in descending
    # order by recall.
    prec_rec = []
    while (len(detected_bboxes) > 0):
        # Get precision recall with 0.5 IoU threshold .
        precision, recall = get_precision_recall_at_iou(
            ground_bboxes, detected_bboxes, 0.5)
        prec_rec.append((precision, recall))
        detected_bboxes.pop()
    # Append (0, 0) to also cover the area from first recall point to 0 recall.
    prec_rec.append((0, 0))
    avg_precision = 0.0
    max_precision = 0.0
    max_precision = None
    last_recall = None
    for (precision, recall) in prec_rec:
        if max_precision is None:
            max_precision = precision
            last_recall = recall
        else:
            avg_precision += (last_recall - recall) * max_precision
            max_precision = max(max_precision, precision)
            last_recall = recall
    return avg_precision


def visualize_no_colors_bboxes(op_name, timestamp, image_np, bboxes):
    add_timestamp(timestamp, image_np)
    for corners in bboxes:
        (xmin, xmax, ymin, ymax) = corners
        color = [128, 0, 0]
        # Show bounding box.
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.imshow(op_name, image_np)
    cv2.waitKey(1)


def visualize_ground_bboxes(op_name, timestamp, image_np, det_objs):
    add_timestamp(timestamp, image_np)
    for det_obj in det_objs:
        det_obj.visualize_on_img(image_np, GROUND_COLOR_MAP)
    cv2.imshow(op_name, image_np)
    cv2.waitKey(1)


def annotate_image_with_bboxes(
        timestamp, image_np, detected_objs, bbox_color_map):
    """ Adds bounding boxes to an image."""
#    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    add_timestamp(timestamp, image_np)
    for detected_obj in detected_objs:
        detected_obj.visualize_on_img(image_np, bbox_color_map)
    return image_np


def visualize_image(op_name, image_np):
    cv2.imshow(op_name, image_np)
    cv2.waitKey(1)


def save_image(image_np, timestamp, data_path, file_base):
    """ Write image to disk."""
    file_name = '{}{}-{}.png'.format(
        data_path,
        file_base,
        timestamp.coordinates[0])
    rgb_img = Image.fromarray(image_np)
    rgb_img.save(file_name)


def visualize_bboxes(
        op_name, timestamp, image_np, detected_objs, bbox_color_map):
#    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    add_timestamp(timestamp, image_np)
    for detected_obj in detected_objs:
        detected_obj.visualize_on_img(image_np, bbox_color_map)
    cv2.imshow(op_name, image_np)
    cv2.waitKey(1)

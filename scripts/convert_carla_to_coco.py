"""
Takes in a folder of observations (images, bounding box annotations) and
 converts them into COCO format (which can be used by e.g. CenterNet).

Example usage:
python convert_carla_to_coco.py --data sample_data --out annotations.json
"""

import json
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data", "data", "Path to data folder.")
flags.DEFINE_string("out", "annotations.json", "Path for output file.")


def _bbox_to_coco_bbox(bbox):
    return [(bbox[0][0]), (bbox[0][1]), (bbox[1][0] - bbox[0][0]),
            (bbox[1][1] - bbox[0][1])]


def main(_):
    categories = ['person', 'vehicle', 'traffic sign']
    cat_ids = {cat: i + 1 for i, cat in enumerate(categories)}

    cat_info = []
    for i, cat in enumerate(categories):
        cat_info.append({'name': cat, 'id': i + 1})

    ret = {'images': [], 'annotations': [], 'categories': cat_info}

    # Get list of indices using bounding box file names.
    idxs = [
        int(s[7:-5]) for s in os.listdir(FLAGS.data) if s.startswith("bboxes")
    ]
    print("COUNT:", len(idxs))
    for idx in idxs:
        image_info = {'file_name': '{}.png'.format(idx), "id": int(idx)}
        ret['images'].append(image_info)
        bbox_path = 'bboxes-{}.json'.format(idx)
        bbox_data = json.load(open(FLAGS.data + "/" + bbox_path, 'r'))
        for entry in bbox_data:
            ann = {
                'image_id': idx,
                'id': int(len(ret['annotations']) + 1),
                'category_id': cat_ids[entry[0]],
                'bbox': _bbox_to_coco_bbox(entry[1]),
            }
            ret['annotations'].append(ann)
    json.dump(ret, open(FLAGS.out, 'w'))


if __name__ == "__main__":
    app.run(main)

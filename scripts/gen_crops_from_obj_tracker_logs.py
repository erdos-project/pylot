"""
Takes in a folder of observations (center-[timestep].png images,
mot-[timestep].txt tracker logs) and creates and saves crops of the bounding
boxes. Useful for training the feature extractor model for DeepSORT tracker.

Example usage:
python gen_crops_from_obj_tracker_logs.py --data sample_data --out crops
"""

import cv2
import glob
import json
import numpy as np
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("bbox_json", False,
                  "If True, generate crops from bbox json log files \
                   instead of mot text log files.")
flags.DEFINE_string("data", "data", "Path to data folder.")
flags.DEFINE_string("out", "crops", "Path to dir for output data.")


def merge_bbox_json_files(dir_path, result_file="combined_logs.txt"):
    """
    Merges individual bbox json files into one txt file of detections.
    These files contain lists of detections, with each detection being
    a 4-item list with format: [class label, detailed label, id, bbox].
    The bbox has format [[xmin, ymin], [xmax, ymax]].

    Args:
        dir_path: Path to directory that holds bboxes-[timestep].json files.
        result_file: Name of output file containing merged log files.
    """
    merged_mot_text = []
    bbox_json_logs = sorted(glob.glob(dir_path + "/bboxes*"),
                            key=lambda line: int(line.split("bboxes-")[1][:-5])) # sort by timestep
    print("Found {} bbox json files.".format(len(bbox_json_logs)))
    for log in bbox_json_logs:
        timestamp = log.split("bboxes-")[1][:-5]
        with open(log, "r") as f:
            data = json.load(f)
            for detection in data:
                general_label, detailed_label, obj_id, bbox_coords = detection
                obj_id = "+".join([detailed_label, str(obj_id)])
                x, y = bbox_coords[0]
                w, h = bbox_coords[1][0] - x, bbox_coords[1][1] - y
                mot_text_line = ",".join([timestamp, obj_id, str(x), str(y), str(w), str(h)])
                merged_mot_text.append(mot_text_line)
    with open(result_file, "w") as f:
        f.write('\n'.join(merged_mot_text))

def merge_mot_txt_files(dir_path, result_file="combined_logs.txt"):
    """
    Merges individual mot-format txt files into one txt file of detections.

    Args:
        dir_path: Path to directory that holds mot-[timestep].txt files.
        result_file: Name of output file containing merged log files.
    """
    relevant_files = sorted(
        glob.glob(dir_path + "/*.txt"),
        key=lambda line: int(line.split("mot-")[1][:-4]))  # sort by timestamp
    print("Found {} mot txt files.".format(len(relevant_files)))
    with open(result_file, "w") as combined_file:
        combined_text = []
        for f in relevant_files:
            with open(f, "r") as sub_file:
                combined_text.extend(sub_file.readlines())
        combined_file.writelines(combined_text)


def get_crops(mot_detections_file, imgs_path, out_dir, area_tol=1500):
    """
    Uses detections/bboxes from mot_detections_file to crop bboxes from
    corresponding images at imgs_path. Only saves crops with area > area_tol.
    Writes new log file that is filtered to only contain the saved crops.
    """
    with open(mot_detections_file, "r") as f:
        mot_data = f.readlines()
    kept_crops_infos = []
    for line in mot_data:
        info = line.split(",")
        timestamp, obj_id, x, y, w, h = info[0], info[1], int(info[2]), int(
            info[3]), int(info[4]), int(info[5])
        img = cv2.imread(imgs_path + "/center-{}.png".format(timestamp))
        crop = img[y:y + h, x:x + w, :]
        if h * w >= area_tol:
            cv2.imwrite(out_dir + "/crop-{}-{}.png".format(timestamp, obj_id),
                        crop)
            kept_crops_infos.append(line)
    print("Keeping {}/{} crops with area >= {}".format(len(kept_crops_infos),
                                                       len(mot_data),
                                                       area_tol))
    with open(out_dir + "/filtered_crops_logs.txt", "w") as f:
        f.writelines(kept_crops_infos)


def convert_crops_to_pytorch_imagefolder_structure(crops_dir):
    """
    Converts crops to training and test set (~90/10 split).
    All crops for a certain object id are grouped into 1 directory.
    (i.e., crops/train/object_id/all_crops_of_this_object_id.png)
    """
    files = glob.glob(crops_dir + "/*.png")
    obj_id_to_crops = {}
    for f in files:
        obj_id = f.split(".png")[0].split("-")[-1]
        if obj_id not in obj_id_to_crops:
            obj_id_to_crops[obj_id] = {f}
        else:
            obj_id_to_crops[obj_id].add(f)
    os.mkdir(crops_dir + "/train")
    os.mkdir(crops_dir + "/test")
    for obj_id in obj_id_to_crops:
        os.mkdir(crops_dir + "/train/" + obj_id)
        os.mkdir(crops_dir + "/test/" + obj_id)
        for f in obj_id_to_crops[obj_id]:
            img_name = f.split("/")[-1]
            if np.random.randint(0, 10):
                os.rename(f, crops_dir + "/train/" + obj_id + "/" + img_name)
            else:
                os.rename(f, crops_dir + "/test/" + obj_id + "/" + img_name)


def main(_):
    log_file_path = FLAGS.data + "/combined_logs.txt"
    if FLAGS.bbox_json:
        merge_bbox_json_files(FLAGS.data, result_file=log_file_path)
    else:
        merge_mot_txt_files(FLAGS.data, result_file=log_file_path)
    get_crops(log_file_path, FLAGS.data, FLAGS.out)
    convert_crops_to_pytorch_imagefolder_structure(FLAGS.out)


if __name__ == "__main__":
    app.run(main)

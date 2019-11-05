import cv2
import glob
import numpy as np
import os


def merge_deepsort_txt_files(dir_path, result_file="combined_logs.txt"):
    """
    Merges individual deepsort txt files into one large txt file of detections

    dir_path: path to directory that holds several deepsort-timestep.txt files
    result_file: name of output file containing merged log files
    """
    relevant_files = sorted(glob.glob(dir_path + "*.txt"), 
                            key=lambda line: int(line.split("deepsort-")[1][:-4])) # sort by timestamp
    with open(result_file, "w") as combined_file:
        combined_text = []
        for f in relevant_files:
            with open(f, "r") as sub_file:
                combined_text.extend(sub_file.readlines())
        combined_file.writelines(combined_text)

def get_crops(deepsort_file, imgs_path, area_tol=1500):
    """
    Uses detections/bboxes from deepsort_file to crop bboxes from corresponding
    images at imgs_path. Only saves crops that have area > area_tol. Writes
    new deepsort log file that is filtered to only contain the saved crops.
    """
    with open(deepsort_file, "r") as f:
        deepsort_data = f.readlines()
    kept_crops_infos = []
    for line in deepsort_data:
        info = line.split(",")
        timestamp, obj_id, x, y, w, h = info[0], info[1], int(info[2]), int(info[3]), int(info[4]), int(info[5])
        img = cv2.imread(imgs_path + "/carla-center-{}.png".format(timestamp))
        crop = img[y:y+h, x:x+w, :]
        if h * w >= area_tol:
            cv2.imwrite("/Users/amandhar/Desktop/crops/crop-{}-{}.png".format(timestamp, obj_id), crop)
            kept_crops_infos.append(line)
    with open(deepsort_file[:-4] + "_filtered.txt", "w") as f:
        f.writelines(kept_crops_infos)

def convert_crops_to_pytorch_imagefolder_structure(crops_dir):
    """
    Converts crops to training and test set (~90/10 split).
    All crops for a certain object id are grouped into 1 directory.
    (so training set looks like processed/object_id/all_crops_of_this_object_id.png)
    """
    files = glob.glob(crops_dir + "/*.png")
    obj_id_to_crops = {}
    for f in files:
        obj_id = f.split(".png")[0].split("-")[-1]
        if obj_id not in obj_id_to_crops:
            obj_id_to_crops[obj_id] = {f}
        else:
            obj_id_to_crops[obj_id].add(f)

    for obj_id in obj_id_to_crops:
        os.mkdir(crops_dir + "_processed/" + obj_id)
        os.mkdir(crops_dir + "_processed_test/" + obj_id)
        for f in obj_id_to_crops[obj_id]:
            img_name = f.split("/")[-1]
            if np.random.randint(0, 10):
                os.rename(f, crops_dir + "_processed/" + obj_id + "/" + img_name)
            else:
                os.rename(f, crops_dir + "_processed_test/" + obj_id + "/" + img_name)


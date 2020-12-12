from absl import app, flags
import os
from pathlib import Path
import re
FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'data/', 'Main path where data is logged, not including town and start')
flags.DEFINE_string('output_path', 'data/lanenet/', 'Path where to place lanenet structured data')
flags.DEFINE_string('town_start', '', 'Town and start where images were collected from')
def main(argv):
    data_path = os.path.join(FLAGS.data_path, FLAGS.town_start)
    data_dir = os.path.abspath(data_path)
    output_dir = os.path.abspath(FLAGS.output_path)
    binary_img_dir = os.path.join(output_dir, "gt_binary_image")
    instance_img_dir = os.path.join(output_dir, "gt_instance_image")
    img_dir = os.path.join(output_dir, "gt_image")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(binary_img_dir).mkdir(parents=True, exist_ok=True)
    Path(instance_img_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    group_files = {}
    rename = {}
    i, j, k = 0, 0, 0
    for f in os.listdir(data_dir):
        curr_path = os.path.join(data_dir, f)
        m = re.search(r"-(\d+)\.", f)
        timestamp = m.group(1)
        new_f = FLAGS.town_start + "_" + timestamp + ".png" if FLAGS.town_start else timestamp +".png"
        if re.search(r"binary_lane-.+\.png", f):
            new_fname = os.path.join(binary_img_dir, new_f)
            i+=1
        elif re.search(r"lane-.+\.png", f):
            new_fname = os.path.join(instance_img_dir, new_f)
            j+=1
        elif re.search(r"center-.+\.png", f):
            new_fname = os.path.join(img_dir, new_f)
            k+=1
        rename[new_fname] = curr_path
        if timestamp in group_files:
            group_files[timestamp].append(new_fname)
        else:
            group_files[timestamp] = [new_fname]
    print(f"{i} binary {j} lane  {k} center")
    with open(os.path.join(output_dir,"im_names.txt"), "a") as f:
        i = 0
        for v in group_files.values():
            if len(v) == 3:
                v_sort = sorted(v) #binary, center, lane
                v_join = [v_sort[1], v_sort[0], v_sort[2]]
                f.write(" ".join(v_join) +'\n')
                for new_fname in v:
                    curr_path = rename[new_fname]
                    Path(curr_path).rename(new_fname)
                i+=1
        print(f"{i} data points ")
                
        
if __name__ == "__main__":
    app.run(main)
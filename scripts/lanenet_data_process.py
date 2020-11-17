from absl import app, flags
import os
from pathlib import Path
import re
FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'data/', 'Path where data is logged')
flags.DEFINE_string('output_path', 'data/lanenet/', 'Path where to place lanenet structured data')

def main(argv):
    data_dir = os.path.abspath(FLAGS.data_path)
    output_dir = os.path.abspath(FLAGS.output_path)
    binary_img_dir = os.path.join(output_dir, "gt_binary_image")
    instance_img_dir = os.path.join(output_dir, "gt_instance_image")
    img_dir = os.path.join(output_dir, "gt_image")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(binary_img_dir).mkdir(parents=True, exist_ok=True)
    Path(instance_img_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(data_dir):
        curr_path = os.path.join(data_dir, f)
        if re.search(f"binary-lane-.+\.png", f):
            Path(curr_path).rename(os.path.join(binary_img_dir, f))
        elif re.search(f"lane-.+\.png", f):
            Path(curr_path).rename(os.path.join(instance_img_dir, f))
        elif re.search(f"carla-center-.+\.png", f):
            Path(curr_path).rename(os.path.join(img_dir, f))
        
if __name__ == "__main__":
    app.run(main)
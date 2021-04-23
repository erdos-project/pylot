from absl import app, flags
from pathlib import Path
import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
import math
from random import sample
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data/', 'binary images directory')

def main(argv):
    data_dir = os.path.abspath(FLAGS.data_path)
    res = []
    # ct = []
    for f in os.listdir(data_dir):
        curr_path = os.path.join(data_dir, f)
        image = Image.open(curr_path)
        im = np.asarray(image)
        num_nonzero = np.count_nonzero(im)
        # ct.append(num_nonzero)
        if num_nonzero < 3000:
            res.append(curr_path)
    # n = math.ceil((max(ct) - min(ct))/1000)
    # plt.hist(ct, bins=n)
    # plt.show()
    print(len(res))
    res_sample = res
    directories = ['gt_binary_image','gt_instance_image', 'gt_image']
    filtered_dir = os.path.join(data_dir[:-len('gt_binary_image')], 'filtered/')
    Path(filtered_dir).mkdir(parents=True, exist_ok=True)
    for d in directories:
        res_dir = os.path.join(filtered_dir, d)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    for f in res_sample:
        s = f.split('gt_binary_image')
        for d in directories:
            curr_dir = d.join(s)
            res_dir = os.path.join(filtered_dir, d) + s[1]
            # print(curr_dir, res_dir)
            Path(curr_dir).rename(res_dir)



if __name__ == "__main__":
    app.run(main)
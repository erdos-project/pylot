'''
Converts images and their corresponding bounding box data to TFRecords,
for use in object detection.

Usage: python create_obj_detection_tf_records.py --input_path input_folder_name --output_path output_folder_name
'''

import tensorflow as tf
import json
import os
import random
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to data folder.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecords.')

FLAGS = flags.FLAGS

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

LABEL_DICT = {
    "person": 1,
    "vehicle": 2,
    "traffic light": 3,
    "stop marking": 4,
    "speed limit": 5
}


def create_tf_example(idx):

    filename = FLAGS.input_path + '/carla-center-' + str(
        idx) + '.png'  # Filename of the image.
    filename = filename.encode()

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'png'.encode()

    xmins = [
    ]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [
    ]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    objs = json.load(
        open(FLAGS.input_path + '/bboxes-' + str(idx) + '.json', 'rb'))
    for label, box in objs:
        if 'traffic light' in label:  # for now, disregard traffic light colors
            label = 'traffic light'
        if 'speed limit' in label:  # for now, disregard values for speed limit
            label = 'speed limit'
        assert 0 <= box[0][0] <= IMAGE_WIDTH and \
               0 <= box[0][1] <= IMAGE_HEIGHT, 'min out of bounds ' + str(box) + ' ' + str(label)
        assert box[0][0] <= box[1][0] <= IMAGE_WIDTH and \
               box[0][1] <= box[1][1] <= IMAGE_HEIGHT, 'max out of bounds ' + str(box) + ' ' + str(label)
        xmins.append(float(box[0][0] / IMAGE_WIDTH))
        xmaxs.append(float(box[1][0] / IMAGE_WIDTH))
        ymins.append(float(box[0][1] / IMAGE_HEIGHT))
        ymaxs.append(float(box[1][1] / IMAGE_HEIGHT))
        classes_text.append(label.encode())
        assert label in LABEL_DICT.keys(), label
        classes.append(int(LABEL_DICT[label]))

    tf_example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/height': dataset_util.int64_feature(IMAGE_HEIGHT),
            'image/width': dataset_util.int64_feature(IMAGE_WIDTH),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(
                classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(
                classes),
        }))

    return tf_example


def main(_):

    train_writer = tf.python_io.TFRecordWriter(FLAGS.output_path +
                                               "/train_set.record")
    eval_writer = tf.python_io.TFRecordWriter(FLAGS.output_path +
                                              "/eval_set.record")

    # Looks for files of the form bboxes-*.json
    idxs = [f[7:-5] for f in os.listdir(FLAGS.input_path) \
                if f.startswith('bboxes-') and f.endswith('.json')]
    random.shuffle(idxs)
    print('Total of {} records.'.format(len(idxs)))
    count = 0
    for i in range(len(idxs)):
        idx = idxs[i]
        if i < len(idxs) / 5:
            eval_writer.write(create_tf_example(idx).SerializeToString())
        else:
            train_writer.write(create_tf_example(idx).SerializeToString())
        count += 1

        if count % 1000 == 0:
            print('Written {} records.'.format(count))
    train_writer.close()
    eval_writer.close()


if __name__ == '__main__':
    tf.app.run()

import erdust
import json
import os


class BoundingBoxLoggerOperator(erdust.Operator):
    """ Logs bounding boxes of DetectedObjects received on a stream."""

    def __init__(self, obstacles_stream, name, flags):
        obstacles_stream.add_callback(self.on_detected_objs_msg)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacles_stream):
        return []

    def on_detected_objs_msg(self, msg):
        """ Invoked upon the receipt of a message on the obstacles stream."""
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_frame != 0:
            return
        bboxes = [det_obj.get_bbox_label() for det_obj in msg.detected_objects]
        timestamp = msg.timestamp.coordinates[0]
        # Write the bounding boxes.
        file_name = os.path.join(self._flags.data_path,
                                 'bboxes-{}.json'.format(timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)

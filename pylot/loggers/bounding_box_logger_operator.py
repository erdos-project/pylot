import json

from erdos.op import Op

from pylot.utils import is_obstacles_stream


class BoundingBoxLoggerOp(Op):
    def __init__(self, name, flags):
        super(BoundingBoxLoggerOp, self).__init__(name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(is_obstacles_stream).add_callback(
            BoundingBoxLoggerOp.on_detected_objs_msg)
        return []

    def on_detected_objs_msg(self, msg):
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_frame != 0:
            return
        bboxes = []
        for detected_obj in msg.detected_objects:
            (xmin, xmax, ymin, ymax) = detected_obj.corners
            bboxes.append((detected_obj.label, ((xmin, ymin), (xmax, ymax))))
        timestamp = msg.timestamp.coordinates[0]
        # Write the bounding boxes.
        file_name = '{}bboxes-{}.json'.format(self._flags.data_path, timestamp)
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)

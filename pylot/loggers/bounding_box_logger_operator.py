import json

from erdos.op import Op

from pylot.utils import is_obstacles_stream


class BoundingBoxLoggerOp(Op):
    def __init__(self, name, flags):
        super(BoundingBoxLoggerOp, self).__init__(name)
        self._flags = flags

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(is_obstacles_stream).add_callback(
            BoundingBoxLoggerOp.on_detected_objs_msg)
        return []

    def on_detected_objs_msg(self, msg):
        bboxes = []
        for detected_obj in msg.detected_objects:
            (xmin, xmax, ymin, ymax) = detected_obj.corners
            bboxes.append((detected_obj.label, ((xmin, ymin), (xmax, ymax))))
        # XXX(ionel): This assumes that the timestamps are tuples.
        timestamp = msg.timestamp.coordinates[1]
        # Write the bounding boxes.
        file_name = '{}bboxes-{}.json'.format(self._flags.data_path, timestamp)
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)

import json
  
from erdos.op import Op

from pylot.utils import is_obstacles_stream, create_deep_sort_logs_stream

class DeepSortLoggerOp(Op):
    def __init__(self, name, flags):
        super(DeepSortLoggerOp, self).__init__(name)
        self._flags = flags
        self._msg_cnt = 0
        self._output_stream_name = 'deep_sort_logs'
    
    @staticmethod
    def setup_streams(input_streams):
        # Register a callback on obstacles data stream.
        input_streams.filter(is_obstacles_stream).add_callback(
            DeepSortLoggerOp.on_detected_objs_msg)
        return [create_deep_sort_logs_stream()]


    def on_detected_objs_msg(self, msg):
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_frame != 0:
            return
        bboxes = [det_obj.get_bbox_label() for det_obj in msg.detected_objects]
        timestamp = msg.timestamp.coordinates[0]
        lines = []
        for det_obj in msg.detected_objects:
            label, bbox_corners = det_obj.get_bbox_label()
            if label in {"pedestrian"}:
                (x1, y1), (x2, y2) = bbox_corners
                bbox_x, bbox_y = x1, y1
                bbox_w, bbox_h = x2 - x1, y2 - y1
                log_line = "{},{},{},{},{},{},{},{},{},{}\n".format(
                    timestamp, det_obj.obj_id, bbox_x, bbox_y, bbox_w, bbox_h, 1.0, -1, -1, -1)
                lines.append(log_line)

        # Send the logs.
        output_msg = (msg.timestamp, lines)
        self.get_output_stream(self._output_stream_name).send(output_msg)
        # Send watermark on the output stream because operators do not
        # automatically forward watermarks when they've registed an
        # on completion callback.
        self.get_output_stream(self._output_stream_name)\
            .send(WatermarkMessage(msg.timestamp))

        # Write the data, MOT16 style: https://motchallenge.net/instructions/
        file_name = '{}deepsort-{}.txt'.format(self._flags.data_path, timestamp)
        with open(file_name, 'w') as outfile:
            outfile.writelines(lines)

import erdos
import json
import os


class BoundingBoxLoggerOperator(erdos.Operator):
    """ Logs bounding boxes of obstacles received on a stream."""
    def __init__(self, obstacles_stream, name, flags, log_file_name=None):
        obstacles_stream.add_callback(self.on_obstacles_msg)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacles_stream):
        return []

    def on_obstacles_msg(self, msg):
        """ Invoked upon the receipt of a message on the obstacles stream."""
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        bboxes = [obstacle.get_bbox_label() for obstacle in msg.obstacles]
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        # Write the bounding boxes.
        file_name = os.path.join(self._flags.data_path,
                                 'bboxes-{}.json'.format(timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)

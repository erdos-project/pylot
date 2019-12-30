import erdos
import os


class MultipleObjectTrackerLoggerOperator(erdos.Operator):
    """ Logs tracked obstacles in the MOT16 file format."""
    def __init__(self, obstacles_stream, name, flags, log_file_name=None):
        # Register a callback on obstacles data stream.
        obstacles_stream.add_callback(self.on_detected_objs_msg)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacles_stream):
        return []

    def on_detected_objs_msg(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_frame != 0:
            return
        timestamp = msg.timestamp.coordinates[0]
        lines = []
        for obstacle in msg.obstacles:
            if obstacle.label in {"pedestrian", "vehicle"}:
                lines.append(obstacle.as_mot16_str(timestamp))

        # Write the data, MOT16 style: https://motchallenge.net/instructions/
        file_name = os.path.join(self._flags.data_path,
                                 'mot-{}.txt'.format(timestamp))
        with open(file_name, 'w') as outfile:
            outfile.writelines(lines)

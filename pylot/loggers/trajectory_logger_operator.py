import json

from erdos.op import Op

from pylot.utils import is_ground_tracking_stream


class TrajectoryLoggerOp(Op):
    def __init__(self, name, flags):
        super(TrajectoryLoggerOp, self).__init__(name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(is_ground_tracking_stream).add_callback(
            TrajectoryLoggerOp.on_trajectories_msg)
        return []

    def on_trajectories_msg(self, msg):
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_frame != 0:
            return
        trajectories = [str(trajectory) for trajectory in msg.obj_trajectories]
        timestamp = msg.timestamp.coordinates[0]
        # Write the trajectories.
        file_name = '{}trajectories-{}.json'.format(self._flags.data_path, timestamp)
        with open(file_name, 'w') as outfile:
            json.dump(trajectories, outfile)


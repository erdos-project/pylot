import erdos
import os
import pickle


class DepthCameraLoggerOperator(erdos.Operator):
    def __init__(self,
                 depth_camera_stream,
                 name,
                 flags,
                 filename_prefix,
                 log_file_name=None):
        depth_camera_stream.add_callback(self.on_depth_frame)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._depth_frame_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(depth_camera_stream):
        return []

    def on_depth_frame(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._depth_frame_cnt += 1
        if self._depth_frame_cnt % self._flags.log_every_nth_frame != 0:
            return
        # Write the depth information.
        file_name = os.path.join(
            self._flags.data_path,
            self._filename_prefix + str(msg.timestamp.coordinates[0]) + '.pkl')
        pickle.dump(msg.frame,
                    open(file_name, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

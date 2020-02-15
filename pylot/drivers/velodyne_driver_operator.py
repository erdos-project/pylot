import erdos


class VelodyneDriverOperator(erdos.Operator):
    def __init__(self,
                 point_cloud_stream,
                 name,
                 lidar_setup,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        self._point_cloud_stream = point_cloud_stream
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def run(self):
        pass

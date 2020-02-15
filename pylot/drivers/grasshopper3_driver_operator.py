import erdos


class Grasshopper3DriverOperator(erdos.Operator):
    def __init__(self,
                 camera_stream,
                 name,
                 camera_setup,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        self._camera_stream = camera_stream
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

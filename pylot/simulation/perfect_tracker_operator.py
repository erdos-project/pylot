from collections import defaultdict, deque
import threading

from erdos.data_stream import DataStream
from erdos.message import WatermarkMessage
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.perception.messages import ObjTrajectory, ObjTrajectoriesMessage

class PerfectTrackerOp(Op):
    """Operator that gives past trajectories of other agents in
       the environment, i.e. their past (x,y,z) locations from an
       ego-vehicle perspective.
    """

    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """Initializes the PerfectTracker Operator.
        """
        super(PerfectTrackerOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._output_stream_name = output_stream_name

        # Queues of incoming data.
        self._vehicles_raw_msgs = deque()
        self._pedestrians_raw_msgs = deque()
        self._can_bus_msgs = deque()

        # Processed data. Key is actor id, value is deque containing the past
        # trajectory of the corresponding actor. Trajectory is stored in world
        # coordinates, for ease of transformation.
        trajectory = lambda: deque(maxlen=self._flags.perfect_tracking_num_steps)
        self._vehicles = defaultdict(trajectory)
        self._pedestrians = defaultdict(trajectory)

        self._lock = threading.Lock()
        self._frame_cnt = 0

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        # Register a callback on vehicles data stream.
        input_streams.filter(pylot.utils.is_ground_vehicles_stream).add_callback(
            PerfectTrackerOp.on_vehicles_update)
        # Register a callback on pedestrians data stream.
        input_streams.filter(pylot.utils.is_ground_pedestrians_stream).add_callback(
            PerfectTrackerOp.on_pedestrians_update)
        # Register a callback on canbus data stream.
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PerfectTrackerOp.on_can_bus_update)
        input_streams.add_completion_callback(PerfectTrackerOp.on_notification)
        return [pylot.utils.create_ground_tracking_stream(output_stream_name)]

    def synchronize_msg_buffers(self, timestamp, buffers):
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    [self._vehicles_raw_msgs, self._pedestrians_raw_msgs,
                     self._can_bus_msgs]):
                return
            vehicles_msg = self._vehicles_raw_msgs.popleft()
            pedestrians_msg = self._pedestrians_raw_msgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()

        self._logger.info('Timestamps {} {} {}'.format(
            vehicles_msg.timestamp, pedestrians_msg.timestamp,
            can_bus_msg.timestamp))

        # The popper messages should have the same timestamp.
        assert (vehicles_msg.timestamp == pedestrians_msg.timestamp ==
                can_bus_msg.timestamp)

        self._frame_cnt += 1

        # Use the most recent can_bus message to convert the past frames
        # of vehicles and pedestrians to our current perspective.

        inv_can_bus_transform = can_bus_msg.data.transform.inverse_transform()

        vehicle_trajectories = []
        # Only consider vehicles which still exist at the most recent timestamp.
        for vehicle in vehicles_msg.vehicles:
            self._vehicles[vehicle.id].append(vehicle)
            cur_vehicle_trajectory = []
            # Iterate through past frames of this vehicle.
            for past_vehicle_loc in self._vehicles[vehicle.id]:
                # Get the location of the center of the vehicle's bounding box,
                # in relation to the CanBus measurement.
                new_transform = inv_can_bus_transform * \
                                past_vehicle_loc.transform * \
                                past_vehicle_loc.bounding_box.transform
                #print (vehicle.id, str(new_transform.location))
                cur_vehicle_trajectory.append(new_transform.location)
            vehicle_trajectories.append(ObjTrajectory('vehicle',
                                                      vehicle.id,
                                                      cur_vehicle_trajectory))

        pedestrian_trajectories = []
        # Only consider pedestrians which still exist at the most recent timestamp.
        for ped in pedestrians_msg.pedestrians:
            self._pedestrians[ped.id].append(ped)
            cur_ped_trajectory = []
            # Iterate through past frames for this pedestrian.
            for past_ped_loc in self._pedestrians[ped.id]:
                # Get the location of the center of the pedestrian's bounding box,
                # in relation to the CanBus measurement.
                new_transform = inv_can_bus_transform * \
                                past_ped_loc.transform * \
                                past_ped_loc.bounding_box.transform
                cur_ped_trajectory.append(new_transform.location)
            pedestrian_trajectories.append(ObjTrajectory('pedestrian',
                                                         ped.id,
                                                         cur_ped_trajectory))

        output_msg = ObjTrajectoriesMessage(vehicle_trajectories + pedestrian_trajectories,
                                            msg.timestamp)

        self.get_output_stream(self._output_stream_name).send(output_msg)
        self.get_output_stream(self._output_stream_name)\
            .send(WatermarkMessage(msg.timestamp))

    def on_vehicles_update(self, msg):
        with self._lock:
            self._vehicles_raw_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        with self._lock:
            self._pedestrians_raw_msgs.append(msg)

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def execute(self):
        self.spin()


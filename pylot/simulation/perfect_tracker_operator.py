from collections import defaultdict, deque
import erdust

from pylot.perception.messages import ObjTrajectory, ObjTrajectoriesMessage


class PerfectTrackerOperator(erdust.Operator):
    """Operator that gives past trajectories of other agents in
       the environment, i.e. their past (x,y,z) locations from an
       ego-vehicle perspective.
    """

    def __init__(self,
                 ground_vehicles_stream,
                 ground_pedestrians_stream,
                 can_bus_stream,
                 ground_tracking_stream,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """Initializes the PerfectTracker Operator. """
        ground_vehicles_stream.add_callback(self.on_vehicles_update)
        ground_pedestrians_stream.add_callback(self.on_pedestrians_update)
        can_bus_stream.add_callback(self.on_can_bus_update)
        erdust.add_watermark_callback([ground_vehicles_stream,
                                       ground_pedestrians_stream,
                                       can_bus_stream]
                                      [ground_tracking_stream],
                                      self.on_watermark)
        self._name = name
        self._logger = erdust.setup_logging(name, log_file_name)
        self._csv_logger = erdust.setup_csv_logging(
            name + '-csv', csv_file_name)
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

    @staticmethod
    def connect(ground_vehicles_stream,
                ground_pedestrians_stream,
                can_bus_stream):
        ground_tracking_stream = erdust.WriteStream()
        return [ground_tracking_stream]

    def on_watermark(self, timestamp, ground_tracking_stream):
        vehicles_msg = self._vehicles_raw_msgs.popleft()
        pedestrians_msg = self._pedestrians_raw_msgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()

        # Use the most recent can_bus message to convert the past frames
        # of vehicles and pedestrians to our current perspective.
        inv_can_bus_transform = can_bus_msg.data.transform.inverse_transform()

        vehicle_trajectories = []
        # Only consider vehicles which still exist at the most recent
        # timestamp.
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
                cur_vehicle_trajectory.append(new_transform.location)
            vehicle_trajectories.append(ObjTrajectory('vehicle',
                                                      vehicle.id,
                                                      cur_vehicle_trajectory))

        pedestrian_trajectories = []
        # Only consider pedestrians which still exist at the most recent
        # timestamp.
        for ped in pedestrians_msg.pedestrians:
            self._pedestrians[ped.id].append(ped)
            cur_ped_trajectory = []
            # Iterate through past frames for this pedestrian.
            for past_ped_loc in self._pedestrians[ped.id]:
                # Get the location of the center of the pedestrian's bounding
                # box, in relation to the CanBus measurement.
                new_transform = inv_can_bus_transform * \
                                past_ped_loc.transform * \
                                past_ped_loc.bounding_box.transform
                cur_ped_trajectory.append(new_transform.location)
            pedestrian_trajectories.append(ObjTrajectory('pedestrian',
                                                         ped.id,
                                                         cur_ped_trajectory))

        output_msg = ObjTrajectoriesMessage(
            timestamp, vehicle_trajectories + pedestrian_trajectories)
        ground_tracking_stream.send(output_msg)

    def on_vehicles_update(self, msg):
        self._vehicles_raw_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        self._pedestrians_raw_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

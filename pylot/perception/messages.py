from erdos.message import Message


class DetectorMessage(Message):
    """ This class represents a detector output message to be sent by
    operators.

    Attributes:
        detected_objects: List of detected objects.
        runtime: Operator runtime (in ms).
    """

    def __init__(self, detected_objects, runtime, timestamp):
        """ Constructs the DetectorMessage with the given data and timestamp.

        Args:
            detected_objects: List of perception.detection.utils.DetectedObject
                objects.
            runtime: Detector operator runtime.
            timestamp: The timestamp of the message.
        """
        super(DetectorMessage, self).__init__(None, timestamp, 'default')
        self.detected_objects = detected_objects
        self.runtime = runtime

    def __str__(self):
        return 'DetectorMessage(timestamp: {}, runtime: {}, '\
            'detected_objects: {})'.format(
                self.timestamp, self.runtime, self.detected_objects)

class SegmentedFrameMessage(Message):
    """ This class represents a message to be used to send segmented frames

    Attributes:
        frame: The segmented frame.
        runtime: The runtime of the operator that produced the segmented frame.
        width: the width of the frame.
        height: the height of the frame.
    """

    def __init__(self, frame, runtime, timestamp):
        """ Constructs the SegmentedFrameMessage with the given data and timestamp.

        Args:
            frame: The segmented frame.
            runtime: The runtime of the operator that produced the segmented
                frame.
            timestamp: The timestamp of the message.
        """
        super(SegmentedFrameMessage, self).__init__(None, timestamp, 'default')
        self.frame = frame
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.runtime = runtime

    def __str__(self):
        return 'SegmentedFrameMessage(timestamp {}, width: {}, '\
            'height: {})'.format(self.timestamp, self.width, self.height)


class ObjPositionSpeed(object):

    def __init__(self,
                 obj_id,
                 x,
                 y,
                 speed_x,  # Speed on the x axis.
                 speed_y,  # Speed on the y axis.
                 sigma_x,  # Variance of the x position estimation.
                 sigma_y,  # Variance of the y position estimation.
                 sigma_speed_x,
                 sigma_speed_y):
        self.obj_id = obj_id
        self.x = x
        self.y = y
        self.speed_x = x
        self.speed_y = y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_speed_x = sigma_speed_x
        self.sigma_speed_y = sigma_speed_y

    def __str__(self):
        return 'Object {} position ({}, {}), position variance ({}, {}), '\
            'axis speed ({}, {}), speed variance ({}, {})'.format(
                self.obj_id, self.x, self.y, self.sigma_x, self.sigma_y,
                self.speed_x, self.speed_y, self.sigma_speed_x,
                self.sigma_speed_y)

class ObjPositionsSpeedsMessage(Message):
    """ This class represents a message to be used to send vehicle position and
        speed info.

    Attributes:
        obj_positions_speeds: The 2D positions and speeds of the objects.
    """

    def __init__(self, obj_positions_speeds, timestamp):
        """ Constructs the ObjPositionsSpeedsMessage with the given data and
        timestamp.

        Args:
            obj_positions_speeds: A list of 2D object positions.
            timestamp: The timestamp of the message.
        """
        super(ObjPositionsSpeedsMessage, self).__init__(None, timestamp, 'default')
        self.obj_positions_speeds = obj_positions_speeds

    def __str__(self):
        return 'ObjPositionsSpeedsMessage(timestamp {}, positions: {})'.format(
            self.timestamp, self.obj_positions_speeds)

class ObjTrajectory(object):

    def __init__(self,
                 obj_class,
                 obj_id,
                 trajectory):
        """Constructs the object trajectory using the given data.

        Args:
            obj_class: String for the class of the object.
            obj_id: ID of the object.
            trajectory: List of past locations of the object.
        """

        self.obj_class = obj_class
        self.obj_id = obj_id
        self.trajectory = trajectory

    def __str__(self):
        trajectory_str = ', '.join([str(loc) for loc in self.trajectory])
        return '{} {}, Trajectory {}'.format(self.obj_class,
                                             self.obj_id,
                                             self.trajectory)

class ObjTrajectoriesMessage(Message):
    """ This class represents a message to be used to send vehicle
        trajectory info.
    """

    def __init__(self, obj_trajectories, timestamp):
        """ Constructs the ObjTrajectoriesMessage with the given data and
        timestamp.

        Args:
            obj_trajectories: A list of ObjTrajectory instances.
        """
        super(ObjTrajectoriesMessage, self).__init__(None, timestamp, 'default')
        self.obj_trajectories = obj_trajectories

    def __str__(self):
        trajectories_str = '\n'.join([str(traj) for traj in self.obj_trajectories])
        return 'ObjTrajectoriesMessage(timestamp {}, trajectories:\n{})'.format(
            self.timestamp, trajectories_str)

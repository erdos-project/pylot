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


class DepthFrameMessage(Message):
    """ This class represents a message to be used to send depth frames

    Attributes:
        frame: The depth frame.
        runtime: The runtime of the operator that produced the depth frame.
        width: the width of the frame.
        height: the height of the frame.
    """

    def __init__(self, frame, runtime, timestamp):
        """ Constructs the DepthFrameMessage with the given data and timestamp.

        Args:
            frame: The depth frame.
            runtime: The runtime of the operator that produced the depth frame.
            timestamp: The timestamp of the message.
        """
        super(DepthFrameMessage, self).__init__(None, timestamp, 'default')
        self.frame = frame
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.runtime = runtime

    def __str__(self):
        return 'DepthFrameMessage(timestamp {}, width: {}, '\
            'height: {})'.format(self.timestamp, self.width, self.height)

    
class ObjPosition(object):

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


class ObjPositionsMessage(Message):
    """ This class represents a message to be used to send vehicle position info

    Attributes:
        obj_positions: The 2D positions of the objects.
    """

    def __init__(self, obj_positions, timestamp):
        """ Constructs the ObjPositionsMessage with the given data and
        timestamp.

        Args:
            obj_positions: A list of 2D object positions.
            timestamp: The timestamp of the message.
        """
        super(ObjPositionsMessage, self).__init__(None, timestamp, 'default')
        self.obj_positions = obj_positions

    def __str__(self):
        return 'ObjPositionsMessage(timestamp {}, positions: {})'.format(
            self.timestamp, self.obj_positions)

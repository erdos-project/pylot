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
        return 'DetectorMessage(timestamp: {}, runtime: {}, detected_objects: {})'.format(
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
        return 'SegmentedFrameMessage(timestamp {}, width: {}, height: {})'.format(
            self.timestamp, self.width, self.height)


class ObjPositionsMessage(Message):
    """ This class represents a message to be used to send segmented frames

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

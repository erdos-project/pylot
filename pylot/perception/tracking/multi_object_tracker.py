class MultiObjectTracker(object):
    def __init__(self):
        self._trackers = []

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame: perception.camera_frame.CameraFrame to reinitialize with.
            obstacles: List of perception.detection.obstacle.Obstacle.
        """
        pass

    def track(self, frame):
        """ Tracks obstacles in a frame.

        Args:
            frame: perception.camera_frame.CameraFrame to track in.
        """
        return True, [tracker.track(frame) for tracker in self._trackers]

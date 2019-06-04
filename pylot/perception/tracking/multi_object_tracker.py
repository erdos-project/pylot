class MultiObjectTracker(object):
    def __init__(self):
        self._trackers = []

    def reinitialize(self, frame, bboxes):
        pass

    def track(self, frame):
        return True, [tracker.track(frame) for tracker in self._trackers]

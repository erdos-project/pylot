import numpy as np
import torch

from nanonets_object_tracking.deepsort import deepsort_rbc
#from DaSiamRPN.code.net import SiamRPNvot
#from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track

from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags, logger):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(wt_path="/home/erdos/workspace/forks/pylot/pylot/perception/tracking/nanonets_object_tracking/ped_feature_extractor")
        self.tracker = None

    def reinitialize(self, frame, bboxes, deep_sort_logs):
        detections, scores = self.get_detections_and_scores_from_logs(deep_sort_logs)
        self.track(frame, out_scores=scores, detections=detections)


    def track(self, frame, out_scores=None, detections=None):
        print("CALLED TRACK, detections? {}".format(detections)) # detections should be in form x, y, w, h
        if detections:
            self.tracker, detections_class = self._deepsort.run_deep_sort(frame, out_scores, detections) # get out_scores and detections?
        if self.tracker:
            bboxes = []
            for track in self.tracker.tracks:
                print("looking through tracks")
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr() # returns np array of 4 nums (top left and bottom right corner coords)
                id_num = str(track.track_id) # what should I do with this?
                bboxes.append((bbox, id_num))
            print("with detector?: {} tracked these bboxes: {}".format(detections, bboxes))
            return True, bboxes
        return False, []


    def get_detections_and_scores_from_logs(self, deep_sort_logs):
        detections, scores = [], []
        #logs_list = list(map(lambda x:x.strip(), deep_sort_logs))
        for i in range(len(deep_sort_logs)):
            info = list(deep_sort_logs[i].split(','))
            info = list(map(float,info))

            coords = list(info[2:6])
            bbox = [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]]
            confidence = info[6]
            detections.append(coords)
            scores.append(confidence)
        return list(detections), scores


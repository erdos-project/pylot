import numpy as np
import torch

from pylot.dependencies.nanonets_object_tracking.deepsort import *

#from DaSiamRPN.code.net import SiamRPNvot
#from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track

from pylot.perception.tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self, flags):
        # Initialize the deepsort object, which has a tracker object within it
        self._deepsort = deepsort_rbc(wt_path=flags.deepsort_siam_model_path)

    def reinitialize(self, frame, bboxes, deep_sort_logs):
        detections, scores = get_detections_and_scores_from_logs(deep_sort_logs)
        self.track(frame, out_scores=scores, detections=detections)


    def track(self, frame, out_scores=None, detections=None):
        tracker, detections_class = self._deepsort.run_deep_sort(frame, out_scores, detections) # get out_scores and detections?
        bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr() # returns np array of 4 nums (upper left and lower right corner coords)
            id_num = str(track.track_id) # what should I do with this?
        bboxes.append(bbox)
        return True, bboxes


    def get_detections_and_scores_from_logs(deep_sort_logs):
        detections, scores = [], []
        logs_list = list(map(lambda x:x.strip(), deep_sort_logs))
        for i in range(len(logs_list)):
            info = list(logs_list[i].split(','))
            info = list(map(float,info))

            coords = info[2:6]
            confidence = info[6]
            detections.append(coords)
            scores.append(confidence)
        return detections, scores



#    def get_dict_from_deep_sort_logs(deep_sort_logs):
#        d = list(map(lambda x:x.strip(), deep_sort_logs))
#        last_frame = int(d[-1].split(',')[0])
#        gt_dict = {x:[] for x in range(last_frame+1)}
#        for i in range(len(d)):
#            a = list(d[i].split(','))
#            a = list(map(float,a))	
#
#            coords = a[2:6]
#            confidence = a[6]
#            gt_dict[a[0]].append({'coords':coords,'conf':confidence})
#
#        return gt_dict






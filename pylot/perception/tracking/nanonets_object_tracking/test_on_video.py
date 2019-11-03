import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2,pickle,sys

from deepsort import *


def get_gt(image,frame_id,gt_dict):

	if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
		return None,None,None

	frame_info = gt_dict[frame_id]

	detections = []
	ids = []
	out_scores = []
	for i in range(len(frame_info)):

		coords = frame_info[i]['coords']

		x1,y1,w,h = coords
		x2 = x1 + w
		y2 = y1 + h

		xmin = min(x1,x2)
		xmax = max(x1,x2)
		ymin = min(y1,y2)
		ymax = max(y1,y2)	

		detections.append([x1,y1,w,h])
		out_scores.append(frame_info[i]['conf'])

	return detections,out_scores


def get_dict(filename):
	with open(filename) as f:	
		d = f.readlines()

	d = list(map(lambda x:x.strip(),d))

	last_frame = int(d[-1].split(',')[0])

	gt_dict = {x:[] for x in range(last_frame+1)}

	for i in range(len(d)):
		a = list(d[i].split(','))
		a = list(map(float,a))	

		coords = a[2:6]
		confidence = a[6]
		gt_dict[a[0]].append({'coords':coords,'conf':confidence})

	return gt_dict

def get_mask(filename):
	mask = cv2.imread(filename,0)
	mask = mask / 255.0
	return mask


if __name__ == '__main__':
	
	#Load detections for the video. Options available: yolo,ssd and mask-rcnn
	filename = 'det/det_ssd512.txt'
	gt_dict = get_dict(filename)

	cap = cv2.VideoCapture('vdo.avi')

	#an optional mask for the given video, to focus on the road. 
	mask = get_mask('roi.jpg')

	#Initialize deep sort.
	deepsort = deepsort_rbc()

	frame_id = 1

	mask = np.expand_dims(mask,2)
	mask = np.repeat(mask,3,2)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('ssd_out_3.avi',fourcc, 10.0, (1920,1080))


	while True:
		print(frame_id)		

		ret,frame = cap.read()
		if ret is False:
			frame_id+=1
			break	

		frame = frame * mask
		frame = frame.astype(np.uint8)

		detections,out_scores = get_gt(frame,frame_id,gt_dict)

		if detections is None:
			print("No dets")
			frame_id+=1
			continue

		detections = np.array(detections)
		out_scores = np.array(out_scores) 

		tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

		for track in tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			
			bbox = track.to_tlbr() #Get the corrected/predicted bounding box
			id_num = str(track.track_id) #Get the ID for the particular track.
			features = track.features #Get the feature vector corresponding to the detection.

			#Draw bbox from tracker.
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
			cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

			#Draw bbox from detector. Just to compare.
			for det in detections_class:
				bbox = det.to_tlbr()
				cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
		
		cv2.imshow('frame',frame)
		out.write(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		frame_id+=1

import glob
import os,sys
import cv2
import numpy as np 


def draw_gt(image,frame_id,gt_dict):

	if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
		return image,None

	frame_info = gt_dict[frame_id]

	crops = []
	ids = []
	for i in range(len(frame_info)):

		coords = frame_info[i]['coords']
		id = frame_info[i]['id']


		x1,y1,w,h = coords
		x2 = x1 + w
		y2 = y1 + h

		xmin = min(x1,x2)
		xmax = max(x1,x2)
		ymin = min(y1,y2)
		ymax = max(y1,y2)	

		crop = image[ymin:ymax,xmin:xmax,:]
		crops.append(crop)
		ids.append(frame_info[i]['id'])

	#crop = image[x1:x2,y1:y2,:]
	#crops = np.array(crops)

	return crops,ids


def get_dict(folder):
	with open(folder+'/gt/gt.txt') as f:
		d = f.readlines()

	d = list(map(lambda x:x.strip(),d))

	last_frame = int(d[-1].split(',')[0])

	gt_dict = {x:[] for x in range(last_frame+1)}

	for i in range(len(d)):
		a = list(d[i].split(','))
		a = list(map(int,a))	

		id = a[1]
		coords = a[2:6]

		gt_dict[a[0]].append({'coords':coords,'id':id})

	return gt_dict

#folders = os.listdir('.')

f = glob.glob('/media/rbc-gpu/DAAA9F78AA9F503D/nvidia-data/track_1/train/*')
folders = []
#print(f)

for i in f:
	if '0' in i:
		g = glob.glob(i+'/*')
		a = [x for x in g if '0' in x.split('/')[-1]]
		if a!=[]:
			folders = folders + a		
			

folders = [x for x in folders if '.py' not in x]


if not os.path.exists('crops/'):
	os.mkdir('crops')


for f in folders:
	d = get_dict(f)
	cap = cv2.VideoCapture(f+'/vdo.avi')

	frame_id = 1

	while True:
		print(frame_id)
		
		ret,frame = cap.read()
		if ret is False:
			break

		crops,vehicle_ids = draw_gt(frame,frame_id,d)
		

		if vehicle_ids is None:
			frame_id+=1
			continue

		for i in range(len(crops)):

			target_dir = 'crops/'+str(vehicle_ids[i])
			
			if not os.path.exists(target_dir):
				os.mkdir(target_dir)

			filename = target_dir + '/'+ f.split('/')[-2] + '_'+ f.split('/')[-1] + '_' + str(frame_id)+'_'+str(vehicle_ids[i])+'.jpg'

			if not os.path.exists(filename):
				cv2.imwrite(filename,crops[i])


		#Finally.	
		frame_id+=1
		



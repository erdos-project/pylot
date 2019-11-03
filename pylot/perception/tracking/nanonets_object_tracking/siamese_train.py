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
import os
from siamese_dataloader import *
from siamese_net import *

import nonechucks as nc
from scipy.stats import multivariate_normal

"""
Get training data
"""

class Config():
	training_dir = "/data/amandhar/crops/"
	testing_dir = "crops_test/"
	train_batch_size = 128
	train_number_epochs = 700	

folder_dataset = dset.ImageFolder(root=Config.training_dir)

transforms = torchvision.transforms.Compose([
	torchvision.transforms.Resize((128,128)),
	torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
	torchvision.transforms.ToTensor()
	])


def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j] #128 is input size.
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask



siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset,transform=transforms,should_invert=False)
net = SiameseNetwork().cuda()

criterion = TripletLoss(margin=1)
optimizer = optim.Adam(net.parameters(),lr = 0.0005 ) #changed from 0.0005

print(torch.cuda.is_available())

counter = []
loss_history = [] 
iteration_number= 0

train_dataloader = DataLoader(siamese_dataset,shuffle=True,num_workers=14,batch_size=Config.train_batch_size)

#Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask().cuda()

for epoch in range(0,Config.train_number_epochs):
	for i, data in enumerate(train_dataloader,0):

		anchor, positive, negative = data
		anchor, positive, negative = anchor.cuda(), positive.cuda() , negative.cuda()
 
		anchor,positive,negative = anchor * gaussian_mask , positive * gaussian_mask, negative * gaussian_mask

		optimizer.zero_grad()

		anchor_out,positive_out,negative_out = net(anchor,positive,negative)

		triplet_loss = criterion(anchor_out,positive_out,negative_out)
		triplet_loss.backward()
		optimizer.step()

		if i %10 == 0 :
			print("Epoch number {}\n Current loss {}\n".format(epoch,triplet_loss.item()))
			iteration_number +=10
			counter.append(iteration_number)
			loss_history.append(triplet_loss.item())
	if epoch%20==0:
		if not os.path.exists('ckpts/'):
			os.mkdir('ckpts')
		torch.save(net,'ckpts/model'+str(epoch)+'.pt')

show_plot(counter,loss_history,path='ckpts/loss.png')

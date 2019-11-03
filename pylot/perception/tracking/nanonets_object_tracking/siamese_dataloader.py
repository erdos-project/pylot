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

from imgaug import augmenters as iaa
import imgaug as ia

import glob 

def imshow(img,text=None,should_save=False):
	npimg = img.numpy()
	plt.axis("off")
	if text:
		plt.text(75, 8, text, style='italic',fontweight='bold',
			bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()    

def show_plot(iteration,loss,path='loss.png'):
	plt.plot(iteration,loss)
	#plt.show()
	plt.savefig(path)



class SiameseTriplet(Dataset):
	
	def __init__(self,imageFolderDataset,transform=None,should_invert=True):
		self.imageFolderDataset = imageFolderDataset    
		self.transform = transform
		self.should_invert = should_invert
		
	def __getitem__(self,index):
		img0_tuple = random.choice(self.imageFolderDataset.imgs)
		
		while True:
			#keep looping till a different class image is found. Negative image.	
			img1_tuple = random.choice(self.imageFolderDataset.imgs) 
			if img0_tuple[1] !=img1_tuple[1]:
				break

		#Selecting positive image.
		anchor_image_name = img0_tuple[0].split('/')[-1]
		anchor_class_name = img0_tuple[0].split('/')[-2]

		all_files_in_class = glob.glob(self.imageFolderDataset.root+anchor_class_name+'/*')
		all_files_in_class = [x for x in all_files_in_class if x!=img0_tuple[0]]
			
		if len(all_files_in_class)==0:
			positive_image = img0_tuple[0]
		else:
			positive_image = random.choice(all_files_in_class)

		if anchor_class_name != positive_image.split('/')[-2]:
			print("Error")


		anchor = Image.open(img0_tuple[0])
		negative = Image.open(img1_tuple[0])
		positive = Image.open(positive_image)

		anchor = anchor.convert("RGB")
		negative = negative.convert("RGB")
		positive = positive.convert("RGB")
		
		if self.should_invert:
			anchor = PIL.ImageOps.invert(anchor)
			positive = PIL.ImageOps.invert(positive)
			negative = PIL.ImageOps.invert(negative)

		if self.transform is not None:
			anchor = self.transform(anchor)
			positive = self.transform(positive)
			negative = self.transform(negative)

		return anchor, positive, negative

	def __len__(self):
		return len(self.imageFolderDataset.imgs)
		
		


class SiameseNetworkDataset(Dataset):
	
	def __init__(self,imageFolderDataset,transform=None,should_invert=True):
		self.imageFolderDataset = imageFolderDataset    
		self.transform = transform
		self.should_invert = should_invert
		
	def __getitem__(self,index):
		img0_tuple = random.choice(self.imageFolderDataset.imgs)
		#we need to make sure approx 50% of images are in the same class
		should_get_same_class = random.randint(0,1) 
		if should_get_same_class:
			while True:
				#keep looping till the same class image is found
				img1_tuple = random.choice(self.imageFolderDataset.imgs) 
				if img0_tuple[1]==img1_tuple[1]:
					break
		else:
			while True:
				#keep looping till a different class image is found
				
				img1_tuple = random.choice(self.imageFolderDataset.imgs) 
				if img0_tuple[1] !=img1_tuple[1]:
					break

		img0 = Image.open(img0_tuple[0])
		img1 = Image.open(img1_tuple[0])
		
		img0 = img0.convert("RGB")
		img1 = img1.convert("RGB")
		
		if self.should_invert:
			img0 = PIL.ImageOps.invert(img0)
			img1 = PIL.ImageOps.invert(img1)

		if self.transform is not None:
			img0 = self.transform(img0)
			img1 = self.transform(img1)
		
		return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
	
	def __len__(self):
		return len(self.imageFolderDataset.imgs)

class ImgAugTransform:
	def __init__(self):
		self.aug = iaa.Sequential([
			iaa.Scale((224, 224)),
			iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
			iaa.Fliplr(0.5),
			iaa.Affine(rotate=(-20, 20), mode='symmetric'),
			iaa.Sometimes(0.25,
						  iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
									 iaa.CoarseDropout(0.1, size_percent=0.5)])),
			iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
		])
	  
	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)


if __name__ == '__main__':

	class Config():
		training_dir = "/data/amandhar/crops/"
		testing_dir = "crops_test/"
		train_batch_size = 64
		train_number_epochs = 100	
	
	folder_dataset = dset.ImageFolder(root=Config.training_dir)

	transforms = torchvision.transforms.Compose([
	torchvision.transforms.Resize((128,128)), #Important. make size= 128
	torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
	torchvision.transforms.ToTensor()
	])

	siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset,transform=transforms,should_invert=False)

	vis_dataloader = DataLoader(siamese_dataset,shuffle=True,num_workers=8,batch_size=1)
	dataiter = iter(vis_dataloader)
	example_batch = next(dataiter)
	concatenated = torch.cat((example_batch[0],example_batch[1],example_batch[2]),0)
	imshow(torchvision.utils.make_grid(concatenated))

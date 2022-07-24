import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode as IMode
from torch.utils.data import DataLoader
from narutils import *
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from utils.imgproc import *
from model import *





class cfg:
	device = torch.device("cuda:0")
	seed=19
	upscale_factor=4
	image_size=512 # hr image size(low res image size=image_size/upscale_factor)
	batch_size=16

	# Train epochs.
	p_epochs=50 # The total number of cycles of the generator training phase.
	epochs=20  # The total number of cycles in the training phase of the adversarial network.


	exp_name='exp-1'
	exp_dir=opj('output',exp_name)
	model_dir=opj(exp_dir,'models')
	output_dir=opj(exp_dir ,'images')
	
	if os.path.exists(exp_dir):
		print('please change exp name')
		sys.exit()

	vgg_path='/work/dataset/vgg-pre.pickle'
	train_path_df=pd.read_csv('path_df.csv').iloc[:5000,:]
	valid_path_df=pd.read_csv('path_df.csv').iloc[5000:10000,:]

	pixel_weight          = 0.01
	content_weight        = 1.0
	adversarial_weight    = 0.001

	# debug=True
	debug=False


	if debug:
		train_path_df=train_path_df.iloc[:8,:]
		valid_path_df=valid_path_df.iloc[:8,:]

class CustomDataset(Dataset):
	def __init__(self,cfg,mode='train'):
		lr_image_size=(cfg.image_size//cfg.upscale_factor,cfg.image_size//cfg.upscale_factor)
		hr_image_size=(cfg.image_size,cfg.image_size)
		self.hr_transforms=transforms.Compose([
			transforms.Resize(hr_image_size,interpolation=IMode.BICUBIC),
		])
		self.lr_transforms=transforms.Compose([
			transforms.Resize(lr_image_size,interpolation=IMode.BICUBIC),
		])
		if mode=='train':
			self.filenames=cfg.train_path_df['file_path'].values
		elif mode=='valid':
			self.filenames=cfg.valid_path_df['file_path'].values
		else:
			raise NotImplementedError

	def __getitem__(self,index):
		hr=image2tensor(cv2.imread(self.filenames[index],-1))
		lr=self.lr_transforms(hr)
		hr=self.hr_transforms(hr)
		return lr,hr

	def __len__(self):
		return len(self.filenames)


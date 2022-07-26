import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
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

class SRGAN:
	def __init__(self,cfg):
		self.cfg=cfg

	def build_dataloader(self,CustomDataset):
		self.train_dataset=CustomDataset(self.cfg,mode='train')
		self.valid_dataset=CustomDataset(self.cfg,mode='valid')

		self.train_loader=DataLoader(self.train_dataset,self.cfg.batch_size,True,pin_memory=True)
		self.valid_loader=DataLoader(self.valid_dataset,self.cfg.batch_size,True,pin_memory=True)

		decopri('successfully built data loaders!')

	def build_model(self):
		self.discriminator=Discriminator().to(self.cfg.device)
		self.generator=Generator().to(self.cfg.device)

		decopri('successfully built models!')

	def train_generator(self,epoch):
		"""Only train the generative model.
		Args:
			train_dataloader (torch.utils.data.DataLoader): The loader of the training data set.
			epoch (int): number of training cycles.

		"""
		# Calculate how many iterations there are under Epoch.
		batches = len(self.train_loader)
		# Put the generative model in training mode.
		self.generator.train()
		pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Train generator|epoch{epoch}',dynamic_ncols=True)
		for index, (lr, hr) in pbar:
			# Copy the data to the specified device.
			lr = lr.to(self.cfg.device)
			hr = hr.to(self.cfg.device)
			# Initialize the gradient of the generated model.
			self.generator.zero_grad()
			# Generate super-resolution images.
			sr = self.generator(lr)
			# Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
			pixel_loss = self.pixel_criterion(sr, hr)
			# Update the weights of the generated model.
			pixel_loss.backward()
			self.p_optimizer.step()


	
	def train_adversarial(self, epoch) :
		"""Training generative models and adversarial models.
		"""
		# Calculate how many iterations there are under Epoch.
		batches = len(self.train_loader)
		# Put the two models in training mode.
		self.discriminator.train()
		self.generator.train()
		pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Train generator|epoch{epoch}',dynamic_ncols=True)
		for index, (lr, hr) in pbar:
			# Copy the data to the specified device.
			lr = lr.to(self.cfg.device)
			hr = hr.to(self.cfg.device)
			label_size = lr.size(0)
			# 打label. Set the real sample label to 1, and the false sample label to 0.
			real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=self.cfg.device)
			fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=self.cfg.device)

			# Initialize the identification model gradient.
			self.discriminator.zero_grad()
			# Generate super-resolution images.
			sr = self.generator(lr)
			# Calculate the loss of the identification model on the high-resolution image.
			hr_output = self.discriminator(hr)

			sr_output = self.discriminator(sr.detach())
			diff=hr_output - torch.mean(sr_output)
			diff[diff<0]=0
			d_loss_hr = self.adversarial_criterion(diff, real_label)
			d_loss_hr.backward()
			d_hr = hr_output.mean().item()
			# Calculate the loss of the identification model on the super-resolution image.
			hr_output = self.discriminator(hr)

			sr_output = self.discriminator(sr.detach())
			diff=sr_output - torch.mean(hr_output)
			diff[diff<0]=0
			d_loss_sr = self.adversarial_criterion(diff, fake_label)
			d_loss_sr.backward()
			d_sr1 = sr_output.mean().item()
			# Update the weights of the authentication model.
			d_loss = d_loss_hr + d_loss_sr
			self.d_optimizer.step()

			# Initialize the gradient of the generated model.
			self.generator.zero_grad()
			# Generate super-resolution images.
			sr = self.generator(lr)
			# Calculate the loss of the identification model on the super-resolution image.
			hr_output = self.discriminator(hr.detach())
			sr_output = self.discriminator(sr)
			# Perceptual loss = 0.01 * pixel loss + 1.0 * content loss + 0.005 * counter loss.
			pixel_loss = self.cfg.pixel_weight * self.pixel_criterion(sr, hr.detach())
			content_loss = self.cfg.content_weight * self.content_criterion(sr, hr.detach())
			diff=sr_output - torch.mean(hr_output)
			diff[diff<0]=0
			adversarial_loss = self.cfg.adversarial_weight * self.adversarial_criterion(diff, real_label)
			# Update the weights of the generated model.
			g_loss = pixel_loss + content_loss + adversarial_loss
			g_loss.backward()
			self.g_optimizer.step()
	

	def train(self):
		self.p_optimizer=optim.Adam(self.generator.parameters(),0.0001,(0.9, 0.999))  # Generate model learning rate during generator training.
		self.d_optimizer=optim.Adam(self.discriminator.parameters(),0.0001,(0.9, 0.999))  # Discriminator learning rate during adversarial network training.
		self.g_optimizer=optim.Adam(self.generator.parameters(),0.0001,(0.9, 0.999))  # The learning rate of the generator during network training.

		# Scheduler.
		self.d_scheduler=StepLR(self.d_optimizer, self.cfg.epochs // 2, 0.1)  # Identify the model scheduler during adversarial training.
		self.g_scheduler=StepLR(self.g_optimizer, self.cfg.epochs // 2, 0.1)
		
		# Loss functions
		self.pixel_criterion=nn.MSELoss().to(self.cfg.device)               # Pixel loss.
		self.content_criterion=ContentLoss(self.cfg).to(self.cfg.device)              # Content loss.
		self.adversarial_criterion=nn.BCELoss().to(self.cfg.device) 

		# train only generator stage
		decopri('Start train generator stage')
		psnr_best=0.0
		for epoch in range(self.cfg.p_epochs):
			self.train_generator(epoch)
			psnr=self.validate(epoch,stage='generator only')
			if (epoch+1)%5==0:
				torch.save(self.generator.state_dict(),opj(self.cfg.model_dir,f'p-{epoch}.pth'))
			if psnr>psnr_best:
				psnr_best=psnr
				mkdirs(self.cfg.model_dir)
				torch.save(self.generator.state_dict(),opj(self.cfg.model_dir,'p-best.pth'))
				print(f'generator saved {opj(self.cfg.model_dir,"p-best.pth")}')
		
		# train adversarial stage
		decopri('Start train adversarial stage')
		self.generator.load_state_dict(torch.load(opj(self.cfg.model_dir,'p-best.pth')))
		for epoch in range(self.cfg.epochs):
			self.train_adversarial(epoch)
			psnr=self.validate(epoch,stage='adversarial')
			if (epoch+1)%5==0:
				torch.save(self.generator.state_dict(),opj(self.cfg.model_dir,f'g-{epoch+1}.pth'))
				torch.save(self.discriminator.state_dict(),opj(self.cfg.model_dir,f'd-{epoch+1}.pth'))
			if psnr>psnr_best:
				psnr_best=psnr
				torch.save(self.generator.state_dict(),opj(self.cfg.model_dir,'g-best.pth'))
				torch.save(self.discriminator.state_dict(),opj(self.cfg.model_dir,'d-best.pth'))

			self.d_scheduler.step()
			self.g_scheduler.step()

	def validate(self,epoch,stage='adversarial'):
		batches=len(self.valid_loader)
		self.generator.eval()
		total_psnr_value=0.0

		with torch.no_grad():
			pbar=tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc=f'valid {stage}|epoch{epoch}',dynamic_ncols=True)
			for index,(lr,hr) in pbar:
				lr = lr.to(self.cfg.device)
				hr = hr.to(self.cfg.device)

				# Generate super-resolution images.
				sr = self.generator(lr)
				if index==0 and (epoch+1)%5==0:
					save_filename=['lr','hr','sr']
					for i,save_img in enumerate([lr,hr,sr]):
						img=(save_img[0].permute(1,2,0).to('cpu').detach().numpy().copy()*255).astype(int)
						mkdirs(self.cfg.output_dir)
						cv2.imwrite(opj(self.cfg.output_dir,f'{stage}-epoch{epoch+1}-{save_filename[i]}.png'),img)

				# Calculate the PSNR indicator.
				mse_loss = ((sr - hr) ** 2).data.mean()
				psnr_value = 10 * torch.log10(1 / mse_loss).item()
				total_psnr_value += psnr_value
			
			avg_psnr_value=total_psnr_value/batches
			print(f'epoch-{epoch} average psnr:{avg_psnr_value}')

		return avg_psnr_value


	def load_model(self,model_path):
		self.generator=Generator().to(self.cfg.device)
		self.generator.load_state_dict(torch.load(model_path,map_location=self.cfg.device))
		decopri('successfully loaded generator!')

	def eval(self,imgs,EvalCustomDataset,verbose=0):
		"""
		input
		imgs:画像リスト
		EvalCustomDataset:eval用のデータセット
		"""
		eval_dataset=EvalCustomDataset(self.cfg,imgs=imgs)
		eval_loader=DataLoader(eval_dataset,self.cfg.eval_batch_size,True,pin_memory=True)
		start=time.time()
		print('eval start')
		for i,lr in enumerate(eval_loader):
			pred=self.generator(lr.to(self.cfg.device))
			if i==0:
				pred_np=pred.permute(0,2,3,1).cpu().detach().numpy()
			else:
				pred_np=np.concatenate([pred_np,pred.permute(0,2,3,1).cpu().detach().numpy()])
		end=time.time()
		if verbose>0:
			print(f'eval end:{end-start:.3f}')
		return pred_np

		
from config import cfg,CustomDataset
from srgan import SRGAN
import shutil
from narutils import *
import argparse
import cv2
import matplotlib.pyplot as plt


def main(args):
	srgan=SRGAN(cfg)
	srgan.load_model(args.model_path)
	img=cv2.imread(args.img_path)
	mkdirs(args.out_dir)
	pred=srgan.eval(img)
	# plt.imsave
	cv2.imwrite(opj(args.out_dir,'res.png'),(pred*255).astype(int))
	print(f'pred img saved {opj(args.out_dir,"res.png")}')

	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path','-I',type=str,default='/work/dataset/images1024x1024/14000/14369.png')
	parser.add_argument('--model_path','-M',type=str,default='./output/exp-1/models/p-best.pth')
	parser.add_argument('--out_dir','-O',type=str,default='./output/exp-1/eval')
	args=parser.parse_args()
	main(args)
	


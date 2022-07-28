from srgan_naru.config import cfg,CustomDataset
from srgan_naru.srgan import SRGAN
import shutil
from narutils import *

def main():
	shutil.copy('./config.py',opj(cfg.exp_dir))
	shutil.copy('./main.py',opj(cfg.exp_dir))
	shutil.copy('./srgan.py',opj(cfg.exp_dir))
	srgan=SRGAN(cfg)
	srgan.build_dataloader(CustomDataset=CustomDataset)
	srgan.build_model()

	srgan.train()

	
if __name__=='__main__':
	main()


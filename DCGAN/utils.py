import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import scipy.misc
import numpy as np

class Logger():
    """ record msg into log file, and print to screen
    """
    def __init__(self, log_file):
        self.log_file = log_file
    def __call__(self, msg):
        with open(self.log_file,'a') as f:
            print(msg, file=f)
            print(msg)

def parser_init():
	parser = argparse.ArgumentParser(description='DCGAN configuartion')
	parser.add_argument('--rm', action='store_true', help='remove the exp folder if exists')
	parser.add_argument('--exp', help='add exp name')
	parser.add_argument('--resume', action='store_true', help='if set truem resume the last training')
	parser.add_argument('--d', help='specify gpu device')

	return parser

def exp_folder_init(args_exp):
	exp_name = args_exp if args_exp else 'default'
	path = './experiments/exp_' + exp_name
	sample_path = path + '/samples'
	params_path = path + '/params'
	log_path = path

	if not os.path.exists(path):
		os.makedirs(path)
		os.makedirs(sample_path)
		os.makedirs(params_path)

	logger = Logger(path + '/exp_' + exp_name + '.csv')

	return logger, sample_path, params_path

def save_multiple_imgs(imgs, path):
	try:
        plt.cla()
	    images = glob.glob(imgs)
	    ind = np.random.choice(len(images), 16)
	    images = [images[i] for i in ind]
	    w, h = 4
	    fig = plt.figure(figsize=(w, h))
	    gs = gridspec.GridSpec(w, h)
	    gs.update(wspace=0.05, hspace=0.05)

	    for i, sample in enumerate(images):
	        sample = scipy.misc.imread(sample)
	        ax = plt.subplot(gs[i])
	        plt.axis('off')
	        ax.set_xticklabels([])
	        ax.set_yticklabels([])
	        ax.set_aspect('equal')
	        plt.imshow(sample.squeeze(), cmap='Greys_r')

	    plt.savefig(save_path)
	    plt.close(fig)
	         
	except:
		print('error in save_multiple_imgs, can not convert arrays to imgs')
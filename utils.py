import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np
import shutil
import math
from scipy.misc import imread
import pylab
import config
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("white")

class Logger():
    """ record msg into log file, and print to screen
    """
    def __init__(self, log_file):
        self.log_file = log_file
    def __call__(self, msg):
        with open(self.log_file,'a') as f:
            #print(msg, file=f)
            print >> f, msg
        print(msg)

def parser_init():
    parser = argparse.ArgumentParser(description='DCGAN configuartion')
    parser.add_argument('--rm', action='store_true', help='remove the exp folder if exists')
    parser.add_argument('--exp', help='add exp name')
    parser.add_argument('--resume', action='store_true', help='if set truem resume the last training')
    parser.add_argument('--dev', help='specify gpu device')
    parser.add_argument('--cat', help='specify category')
    parser.add_argument('--optim', help='specify optim')
    parser.add_argument('--gl', help='gen last layer activation')
    parser.add_argument('--t', help='specify the gan type')
    parser.add_argument('--data', help='specify the dataset')
    parser.add_argument('--ngpu', help='number of GPU', type=int, default=1)
    parser.add_argument('--workers', help='data laoder workers num', type=int, default=1)
    parser.add_argument('--se', help='save epoch', type=int, default=50)
    parser.add_argument('--pe', help='print epoch', type=int, default=25)    
    parser.add_argument('--de', help='decay epoch', type=int, default=200)
    parser.add_argument('--epoch', help='max epoch', type=int, default=200)
    
    return parser

def exp_folder_init(args_exp, epoch = config.MAX_EPOCH, cat='all', resume = False, gan_t='dc', d_set='cifar_10', train=True):
    exp_name = args_exp + '_' if args_exp else ''
    tags = []
    if config.DECAY:
        tags.append('decay')
    if config.SMOOTH_LABEL:
        tags.append('smooth')
    if config.NORM:
        tags.append('norm')
    if config.CLAMP:
        tags.append('clamp')
    if len(tags) > 0:
        exp_name = exp_name + '_'.join(tags) + '_'
     
     
    path = './experiments/' + gan_t +'gan_exp_' + exp_name + d_set
    path += '_' + config.OPTIM + '_' + config.G_LAST 
    path += '_' + cat + '_' + str(epoch)
    sample_path = path
    sample_path = sample_path.replace('experiments','result')
    params_path = path + '/params'
    log_path = path

    if (not resume) and train:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        if os.path.exists(sample_path):
            shutil.rmtree(sample_path)
        os.makedirs(sample_path)
        os.makedirs(params_path)

    logger = Logger(path + '/exp_' + exp_name + '.csv')

    return logger, path, sample_path, params_path

def res_folder_init(args_exp, epoch=config.MAX_EPOCH, cat='all', gan_t='dc', d_set='cifar_10'):
    exp_name = args_exp + '_' if args_exp else ''

    tags = []
    if config.DECAY:
        tags.append('decay')
    if config.SMOOTH_LABEL:
        tags.append('smooth')
    if config.NORM:
        tags.append('norm')
    if config.CLAMP:
        tags.append('clamp')
    if len(tags) > 0:
        exp_name = exp_name + '_'.join(tags) + '_'

    path = './result/' + gan_t +'gan_exp_' + exp_name + d_set
    path += '_' + config.OPTIM + '_' + config.G_LAST 
    path += '_' + cat + '_' + str(epoch)
    sample_path = path + '/samples'
    log_path = path

    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(sample_path):
        shutil.rmtree(sample_path)
    os.makedirs(sample_path)

    logger = Logger(path + '/exp_' + exp_name + '.csv')

    return logger, path, sample_path

def get_latest_model(path, identifier):
    models = glob.glob('{0}/*{1}*'.format(path, identifier))
    epoch = [int(model.split('_')[-1].split('.')[0]) for model in models]
    ind = np.array(epoch).argsort()
    models = [models[i] for i in ind]
    return models[-1]

def get_model(path, identifier, target_epoch):
    models = glob.glob('{0}/*{1}*'.format(path, identifier))
    epoch = [int(model.split('_')[-1].split('.')[0]) for model in models]
    for (i, e) in enumerate(epoch):
        if e == target_epoch:
            return models[i]
    return False

def save_multiple_imgs(imgs, path):
    
    plt.cla()
    imgs = imgs.transpose(0,2,3,1)
    
    if config.G_LAST == 'sigmoid':
        imgs *= 256.0
    elif config.G_LAST == 'tanh':
        imgs *= 128
        imgs += 128

    images = imgs.astype(np.uint8)
    w = 4
    h = w
    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(w, h)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    plt.savefig(path)
    plt.close(fig)


def gaussian_mixture_circle(batchsize, num_cluster=8, scale=1, std=1):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

def gaussian_mixture_double_circle(batchsize, num_cluster=8, scale=1, std=1):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    # Doubles the scale in case of even number
    even_indices = np.argwhere(rand_indices % 2 == 0)
    mean[even_indices] /= 2
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)


def plot_kde(data, dir=None, filename="kde", color="Greens"):
    if dir is None:
        raise Exception()
    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    bg_color  = sns.color_palette(color, n_colors=256)[0]
    ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
    ax.set_axis_bgcolor(bg_color)
    kde = ax.get_figure()
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    kde.savefig("{}/{}.png".format(dir, filename))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
    if dir is None:
        raise Exception()
    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    pylab.savefig("{}/{}.png".format(dir, filename))

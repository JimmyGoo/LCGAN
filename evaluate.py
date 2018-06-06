import os 
import model
import numpy as np
import utils
import time
import config
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, resnet18
from scipy.stats import entropy
#------------------arguments set up--------------------------#
parser = utils.parser_init()
args = parser.parse_args()

assert(args.data and args.cat and args.t)
os.environ["CUDA_VISIBLE_DEVICES"] = args.dev if args.dev else str(0)
#------------------------------------------------------------#

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloaderi
    dataloader = torch.utils.data.DataLoader(imgs.data, batch_size=batch_size)

    # Load inception model
    if args.data == 'cifar':
        inception_model = inception_v3(pretrained=False, transform_input=False).type(dtype)
        inception_path = './pretrained/inception_v3_google-1a9a5a14.pth'
    elif args.data == 'lsun':
        inception_model = resnet18(pretrained=False)
        inception_model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = inception_model.fc.in_features
        inception_model.fc = nn.Linear(num_ftrs, 5)
 
        inception_path = './pretrained/resnet_best.pkl'

    inception_model.load_state_dict(torch.load(inception_path))
    inception_model = inception_model.cuda()
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    if args.data == 'cifar':
        num_classes = 1000
    elif args.data == 'lsun':
        num_classes = 5
    preds = np.zeros((N, num_classes))
    for i, batch in enumerate(dataloader, 0):
        
        batch = batch.type(dtype) 
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def eval(d_set, cat):
    gan_t = args.t
    imgs_num = 500
    
    _, path, _, params_path = utils.exp_folder_init(args.exp, args.epoch, cat, args.resume, gan_t, d_set, train=False)
    if not os.path.exists(path):
        print('exp not found: ' + path)
        return
    expname = args.exp+'_eval' if args.exp else 'eval'
    logger, res_path, sample_path = utils.res_folder_init(expname, args.epoch, cat, gan_t, d_set)
    logger('result folder: ' + res_path)


    ngpu = args.ngpu
    if d_set == 'cifar':
        img_size = config.CIFAR_IMG_SIZE
        ngf = ndf = config.CIFAR_IMG_SIZE
    elif d_set == 'lsun':
        img_size = config.LSUN_IMG_SIZE
        ngf = ndf = config.LSUN_IMG_SIZE
    gen = model.DCGAN_G(img_size, config.Z_SIZE, 3, ngf, ngpu).cuda()
   
    #---only for mixed gen---# 
    
    model_num = int(args.epoch / args.se)
    score_mm = np.zeros((config.EVAL_ITER, model_num))
    score_mstd = np.zeros((config.EVAL_ITER, model_num))
    for i in range(config.EVAL_ITER):
        rand_z = Variable(torch.rand(imgs_num, config.Z_SIZE, 1, 1)).cuda()

        for j in range(model_num):
            target_epoch = (j+1)*args.se
            logger('loding epoch {0} from {1}'.format(target_epoch, params_path))
            gen_params = utils.get_model(params_path, 'gen', target_epoch)
            if gen_params:
                gen.load_state_dict(torch.load(gen_params))
            else:
                logger('epoch {0} params not found at {1}'.format(target_epoch, params_path))
                continue

            imgs = gen(rand_z)
            if args.data == 'cifar':
                score_m, score_std = inception_score(imgs)
            elif args.data == 'lsun':
                score_m, score_std = inception_score(imgs, resize=False)
            score_mm[i, j] = score_m
            score_mstd[i, j] = score_std
        
    for j in range(model_num):  
        is_m = np.sum(score_mm[:,j]) / config.EVAL_ITER
        is_std = np.sum(score_mstd[:, j]) / config.EVAL_ITER
        logger(params_path + ' epoch ' + str((j+1) * args.se))
        logger('score_m: {0} score_std: {1}'.format(is_m, is_std))
    #------------------#


if __name__ == '__main__':
        
    eval(args.data, args.cat)

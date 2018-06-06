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
from torchvision.models import inception_v3
from scipy.stats import entropy
import shutil
#------------------arguments set up--------------------------#
parser = utils.parser_init()
args = parser.parse_args()

assert(args.t)
os.environ["CUDA_VISIBLE_DEVICES"] = args.dev if args.dev else str(0)
#------------------------------------------------------------#


def eval():
    gan_t = args.t

    res_path = './result/' + gan_t +'gan_exp_gaussion_mixed_'
    res_path += str(config.MAX_EPOCH)
    sample_path = res_path + '/samples'
    log_path = res_path

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    if os.path.exists(sample_path):

    
        shutil.rmtree(sample_path)
    os.makedirs(sample_path)

    logger = utils.Logger(res_path + '/exp.csv')
    logger('result folder: ' + res_path)

    z_size = 256
    gaussion_size = 2
    scale = 2.0
    num_mixture = 8
    num_updates_per_epoch = 500    
    batchsize = 100

    gen = model.MLP_G(z_size,gaussion_size).cuda()
    dis = model.MLP_D(gaussion_size).cuda()
    gen_params = list(gen.parameters())
    dis_params = list(dis.parameters())

    dis_optim = optim.Adam(dis_params, lr=config.DIS_LEARNING_RATE, betas=(0.5,0.999))
    gen_optim = optim.Adam(gen_params, lr=config.GEN_LEARNING_RATE, betas=(0.5,0.999))

    if gan_t =='dc':
        criterion = nn.BCELoss()
    elif gan_t == 'ls':
        criterion = nn.MSELoss()
    elif gan_t == 'coop':
        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()

    start_time = time.time()

    alpha = 0
    for current_epoch in xrange(1, args.epoch):

        for t in xrange(num_updates_per_epoch):

            for p in dis.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            dis.zero_grad()
            gen.zero_grad()
            # sample from data distribution
            real = utils.gaussian_mixture_circle(batchsize, num_mixture, scale=scale, std=0.2)
            real = Variable(torch.from_numpy(real)).cuda()
            # sample from generator
            z = Variable(torch.randn(batchsize, z_size)).cuda()
            fake = gen(z)

            real_logit = dis(real / scale)
            fake_logit = dis(fake / scale)
           
            real_labels = Variable(torch.ones(batchsize)).cuda()
            fake_labels = Variable(torch.zeros(batchsize)).cuda()
            #import ipdb
            #ipdb.set_trace()            
 
            if gan_t == 'dc' or gan_t == 'ls':
                dis_loss_real = criterion(real_logit, real_labels)
                dis_loss_fake = criterion(fake_logit, fake_labels)
                dis_loss = dis_loss_fake + dis_loss_real

            elif gan_t == 'coop': 
                dis_loss_real = criterion_bce(real_logit, real_labels)
                dis_loss_fake = criterion_bce(fake_logit, fake_labels)
                dis_loss_bce = alpha * (dis_loss_fake + dis_loss_real)
 
                dis_loss_real = criterion_mse(real_logit, real_labels)
                dis_loss_fake = criterion_mse(fake_logit, fake_labels)
                dis_loss_mse = (1 - alpha) * (dis_loss_fake + dis_loss_real)

                dis_loss = dis_loss_bce + dis_loss_mse
                
            dis_loss.backward()
            dis_optim.step()

            #update gen
            for p in dis.parameters():
                p.requires_grad = False

            gen.zero_grad()
            dis.zero_grad()

            z = Variable(torch.randn(batchsize, z_size)).cuda()
            fake = gen(z)
            fake_logit = dis(fake)
        
            if gan_t == 'dc' or gan_t == 'ls':
                gen_loss_cri = criterion(fake_logit, real_labels)

            elif gan_t == 'coop': 
                gen_loss_bce = alpha * criterion_bce(fake_logit, real_labels)
                gen_loss_mse =  (1 - alpha) * criterion_mse(fake_logit, real_labels)
                gen_loss_cri = gen_loss_bce + gen_loss_mse

            gen_loss = gen_loss_cri

            gen_loss.backward()
            gen_optim.step()

        if current_epoch % args.pe == 0:
            logger('epoch: {0}/{1}'.format(current_epoch, args.epoch))

            if gan_t == 'dc' or gan_t == 'ls':
                logger('gen_loss: {:.4f}, dis_loss: {:.4f}'.format(gen_loss.data[0], dis_loss.data[0]))

            elif gan_t == 'coop': 
                logger('gen_loss: {:.4f}(bce/mse:{:.4f}/{:.4f}), dis_loss: {:.4f}(bce/mse:{:.4f}/{:.4f})'.format(gen_loss.data[0], gen_loss_bce.data[0], gen_loss_mse.data[0], dis_loss.data[0], dis_loss_bce.data[0], dis_loss_mse.data[0]))

            logger('dis logits: real is {:.2f}, fake is {:.2f}'.format(torch.mean(real_logit).data[0], torch.mean(fake_logit).data[0]))
            logger('time spent: {:.4f}'.format(time.time() - start_time))
            start_time = time.time()

        if current_epoch % args.se == 0:

            gen.eval()           
            num_samples = 1000
            samples_true = utils.gaussian_mixture_circle(num_samples, num_cluster=num_mixture, scale=2, std=0.2)
            utils.plot_scatter(samples_true, sample_path, "scatter_true_{0}".format(current_epoch))
            utils.plot_kde(samples_true, sample_path, "kde_true_{0}".format(current_epoch))
            z = Variable(torch.randn(num_samples, z_size)).cuda()
            fake = gen(z)
            utils.plot_scatter(fake.data.cpu().numpy(), sample_path, "scatter_gen_{}".format(current_epoch))
            utils.plot_kde(fake.data.cpu().numpy(), sample_path, "kde_gen_{}".format(current_epoch))

            logger('vis: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))
            gen.train()
    
        
        if gan_t == 'coop' and current_epoch % (args.epoch * config.ALPHA_STEP) == 0:
            alpha = float(current_epoch) / (args.epoch * config.ALPHA_STEP) * config.ALPHA_STEP
            logger('epoch: {0}/{1}, change alpha to: {2}'.format(current_epoch, args.epoch, alpha))


if __name__ == '__main__':
        
    eval()

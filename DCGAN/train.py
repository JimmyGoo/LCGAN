import sys
import os 
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import data
import torch.optim as optim
import argparse
import utils
import time
import torch.autograd.Variable

#------------------arguments set up--------------------------#
parser = parser_init()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.d if arg.d else str(0)
logger, sample_path, params_path = exp_folder_init(args.exp)
#------------------------------------------------------------#


#------------------create components--------------------------#
logger('init components')
gen = Generator(config.Z_SIZE).cuda()
dis = Discriminator().cuda()

gen_params = list(gen.parameters())
dis_params = list(dis.parameters())

current_dis_lr = config.DIS_LEARNING_RATE
current_gen_lr = config.GEN_LEARNING_RATE

dis_optim = optim.Adam(dis_params, lr=config.DIS_LEARNING_RATE, beta=(0.5,0.999))
gen_optim = optim.Adam(gen_params, lr=config.GEN_LEARNING_RATE, beta=(0.5,0.999))

criterion = nn.BCELoss()

logger('init complete')
#------------------------------------------------------------#

def train():
    train_data = CIFAR100Dataset(config.CIFAR_100_TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle = True)

    current_epoch = 1
    start_time = time.time()
    while current_epoch < config.MAX_EPOCH:

        for (i, real_data) in enumerate(train_loader):

            #update dis
            real = Variable(real_data).cuda()
            batch_size = real.size(0)

            for p in dis.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            dis.zero_grad()
            gen.zero_grad()
            
            z = torch.randn(config.BATCH_SIZE, config.Z_SIZE)
            fake = Generator(z)


            fake_logit = Discriminator(fake)
            real_logit = Discriminator(real)

            fake_label = torch.zeros(batch_size)
            real_label = torch.ones(batch_size)

            dis_loss_real = criterion(real_logit, real_labels)
            dis_loss_fake = criterion(fake_logit, fake_labels)
            dis_loss = dis_loss_fake + dis_loss_real

            dis_loss.backward()
            dis_optim.step()

            #update gen
            for p in dis.parameters():
                p.requires_grad = False

            gen.zero_grad()
            dis.zero_grad()

            z = torch.randn(config.BATCH_SIZE, config.Z_SIZE)
            fake = Generator(z)
            fake_logit = Discriminator(fake)
            gen_loss = criterion(fake_logit, real_labels)

            gen_loss.backward()
            gen_optim.step()

            

        if current_epoch % config.PRINT_EPOCH == 0:
            logger('epoch: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))
            logger('gen_loss: {.4f}, dis_loss: {.4f}'.format(gen_loss.data[0], dis_loss.data[0]))
            logger('dis logits: real is {.2f}, fake is {.2f}'.format(torch.mean(real_logit).data[0], torch.mean(fake_logit).data[0]))
            logger('time spent: {.4f}'.format(time.time() - start_time))
            start_time = time.time()

        if current_epoch % config.SAVE_EPOCH == 0:
            logger('save model: {0}/{1}'.format(epoch, config.MAX_EPOCH))
            torch.save(gen.state_dict(), os.path.join(params_path, 'gen_{0}.pkl'.format(current_epoch)))
            torch.save(dis.state_dict(), os.path.join(params_path, 'dis_{0}.pkl'.format(current_epoch)))

            st = time.time()
            gen.eval()
            z = Variable(torch.randn(config.BATCH_SIZE, config.Z_SIZE)).cuda()
            fake = Variable(gen(z))
            save_multiple_imgs(fake.data.cpu().numpy()[:config.NUM_SAVE], sample_path, 'epoch_{}_sample'.format(current_epoch))
            save_multiple_imgs(real_data.data.cpu().numpy()[:config.NUM_SAVE], sample_path, 'epoch_{}_real'.format(current_epoch))
            logger('vis: {0}/{1}, time spent: {2}'.format(current_epoch, config.MAX_EPOCH, time.time() - st))

        if epoch % config.DECAY_LR == 0:
            current_dis_lr = current_dis_lr * 0.5
            current_gen_lr = current_gen_lr * 0.5
            for param_group in gen_optim.param_groups:
                param_group['lr'] = current_gen_lr
            for param_group in dis_optim.param_groups:
                param_group['lr'] = current_dis_lr
            logger('Decay learning rate to gen_lr: {0}, dis_lr: {1}.'.format(current_gen_lr, current_dis_lr))


if __name__ == '__main__':
    train()
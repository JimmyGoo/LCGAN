import os 
import model
import numpy as np
import utils
import time
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
#------------------arguments set up--------------------------#
parser = utils.parser_init()
args = parser.parse_args()
if args.optim:
    config.OPTIM = args.optim
else:
    config.OPTIM = 'adam'

if args.gl:
    config.G_LAST = args.gl
else:
    config.G_LAST = 'tanh'


os.environ["CUDA_VISIBLE_DEVICES"] = args.dev if args.dev else str(1)
#------------------------------------------------------------#

#----------------


def train(d_set, cat):
    alpha = 0
    gan_t = args.t
    config.MAX_EPOCH = args.epoch
    logger, path, sample_path, params_path = utils.exp_folder_init(args.exp, cat, args.resume, gan_t, d_set)
    logger('exp_folder: {}'.format(path))
    logger('res_folder: {}'.format(sample_path))
#------------------create components--------------------------#
    logger('init components')
    
    #if d_set == 'cifar':
    #    gen = model.GeneratorCIFAR(config.Z_SIZE).cuda()
    #    dis = model.DiscriminatorCIFAR().cuda()
    #elif d_set == 'lsun':
    #    gen = model.GeneratorLSUN(config.Z_SIZE).cuda()
    #    dis = model.DiscriminatorLSUN().cuda()
    ngpu = args.ngpu
    if d_set == 'cifar':
        img_size = config.CIFAR_IMG_SIZE
        ngf = ndf = config.CIFAR_IMG_SIZE
    elif d_set == 'lsun':
        img_size = config.LSUN_IMG_SIZE
        ngf = ndf = config.LSUN_IMG_SIZE
    gen = model.DCGAN_G(img_size, config.Z_SIZE, 3, ngf, ngpu).cuda()
    dis = model.DCGAN_D(img_size, config.Z_SIZE, 3, ndf, ngpu).cuda()
    
    gen_params = list(gen.parameters())
    dis_params = list(dis.parameters())

    if config.OPTIM == 'adam':
        dis_optim = optim.Adam(dis_params, lr=config.DIS_LEARNING_RATE, betas=(0.5,0.999))
        gen_optim = optim.Adam(gen_params, lr=config.GEN_LEARNING_RATE, betas=(0.5,0.999))

    elif config.OPTIM == 'rms':
        dis_optim = optim.RMSprop(dis_params, lr=config.DIS_LEARNING_RATE)
        gen_optim = optim.RMSprop(gen_params, lr=config.GEN_LEARNING_RATE)

    else:
        dis_optim = optim.SGD(dis_params, lr=config.DIS_LEARNING_RATE)
        gen_optim = optim.SGD(gen_params, lr=config.GEN_LEARNING_RATE)

    if gan_t == 'dc': 
        criterion = nn.BCELoss()
    elif gan_t == 'ls':
        criterion = nn.MSELoss()
    elif gan_t == 'coop':
        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()

    if args.resume:
        gen_params = utils.get_latest_model(params_path, 'gen')
        gen.load_state_dict(torch.load(gen_params))
        logger('load gen from: {0}'.format(gen_params))
        dis_params = utils.get_latest_model(params_path, 'dis')
        dis.load_state_dict(torch.load(dis_params)) 
        logger('load dis from: {0}'.format(dis_params))

    if config.FIXED_Z:
        fix_z = Variable(torch.randn(config.BATCH_SIZE, config.Z_SIZE)).cuda()

    logger('init complete')
    #------------------------------------------------------------#

    
    current_dis_lr = config.DIS_LEARNING_RATE
    current_gen_lr = config.GEN_LEARNING_RATE
    if d_set == 'cifar':
        if cat != 'mixed':
            train_data = model.CIFAR10Dataset(config.CIFAR_10_PATH, cat)
        else:
            cifar_transform = transforms.Compose([transforms.Resize((config.CIFAR_IMG_SIZE,config.CIFAR_IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_data = torchvision.datasets.CIFAR10(config.CIFAR_10_PATH, train=True, transform=cifar_transform, download=False)
    elif d_set == 'lsun':
        #train_data = model.LSUNDataset(config.LSUN_PATH, cat)
        lsun_transform = transforms.Compose([transforms.Resize((config.LSUN_IMG_SIZE,config.LSUN_IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if cat == 'mixed':
            cat_train = ['bedroom_train', 'kitchen_train', 'restaurant_train', 'dining_room_train', 'living_room_train']
        else:
            cat_train = [cat + '_train']
        train_data = torchvision.datasets.LSUN(config.LSUN_PATH, cat_train, transform=lsun_transform)         

    train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = args.workers)

    current_epoch = 1
    start_time = time.time()
    while current_epoch <= config.MAX_EPOCH:
          
        for (i, b) in enumerate(train_loader): 
            if i >= 10000 and d_set == 'lsun':
                break
            real_data, _ = b
            #update dis
            real = Variable(real_data.float()).cuda()
            batch_size = real.size(0)

            for p in dis.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            dis.zero_grad()
            gen.zero_grad()
           
            z = Variable(torch.randn(batch_size, config.Z_SIZE, 1, 1)).cuda()
            fake = gen(z)
            fake_logit = dis(fake)
            real_logit = dis(real)
            
            if config.SMOOTH_LABEL:
                real_labels = Variable(torch.Tensor(batch_size).uniform_(0.7,1.2)).cuda()
                fake_labels = Variable(torch.Tensor(batch_size).uniform_(0,0.3)).cuda()
            else:
                real_labels = Variable(torch.ones(batch_size)).cuda()
                fake_labels = Variable(torch.zeros(batch_size)).cuda()
            
            #if True or (torch.mean(real_logit).data.cpu().numpy() < 0.85) or (torch.mean(fake_logit).data.cpu().numpy() > 0.75): 
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

            z = Variable(torch.randn(batch_size, config.Z_SIZE, 1, 1)).cuda()
            fake = gen(z)
            fake_logit = dis(fake)
        
            if gan_t == 'dc' or gan_t == 'ls':
                gen_loss_cri = criterion(fake_logit, real_labels)

            elif gan_t == 'coop': 
                gen_loss_bce = alpha * criterion_bce(fake_logit, real_labels)
                gen_loss_mse =  (1 - alpha) * criterion_mse(fake_logit, real_labels)
                gen_loss_cri = gen_loss_bce + gen_loss_mse

            gen_loss_norm =  config.LAMBDA_NORM * (real.norm() - fake.norm()).norm()

            if config.NORM and ((real.norm() - fake.norm()).norm().data[0] > config.NORM_THERSHOLD):
                gen_loss = gen_loss_cri + gen_loss_norm
            else:
                gen_loss = gen_loss_cri
    

            gen_loss.backward()
            gen_optim.step()

        if config.CLAMP:
            for p in dis.parameters():
                p.data.clamp_(config.CLAMP_LOW, config.CLAMP_HIGH)
            for p in gen.parameters():
                p.data.clamp_(config.CLAMP_LOW, config.CLAMP_HIGH)
       
        if current_epoch % args.pe == 0:
            logger('epoch: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))

            if gan_t == 'dc' or gan_t == 'ls':
                logger('gen_loss: {:.4f}(cri/norm:{:.4f}/{:.4f}), dis_loss: {:.4f}'.format(gen_loss.data[0], gen_loss_cri.data[0], gen_loss_norm.data[0], dis_loss.data[0]))

            elif gan_t == 'coop': 
                logger('gen_loss: {:.4f}(bce/mse/norm:{:.4f}/{:.4f}/{:.4f}), dis_loss: {:.4f}(bce/mse:{:.4f}/{:.4f})'.format(gen_loss.data[0], gen_loss_bce.data[0], gen_loss_mse.data[0], gen_loss_norm.data[0], dis_loss.data[0], dis_loss_bce.data[0], dis_loss_mse.data[0]))

            logger('dis logits: real is {:.2f}, fake is {:.2f}'.format(torch.mean(real_logit).data[0], torch.mean(fake_logit).data[0]))
            logger('time spent: {:.4f}'.format(time.time() - start_time))
            start_time = time.time()

        if current_epoch % args.se == 0:
            #import ipdb
            #ipdb.set_trace()
            logger('save model: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))
            torch.save(gen.state_dict(), os.path.join(params_path, 'gen_{0}.pkl'.format(current_epoch)))
            torch.save(dis.state_dict(), os.path.join(params_path, 'dis_{0}.pkl'.format(current_epoch)))

            gen.eval()
        
            if config.FIXED_Z:
                z = fix_z
            else:
                z = Variable(torch.randn(config.BATCH_SIZE, config.Z_SIZE, 1, 1)).cuda()
            fake = Variable(gen(z).data)
            fake.data = fake.data.mul(0.5).add(0.5)
            real.data = real.data.mul(0.5).add(0.5)
            logger('fake norm: {0}, real norm: {1}'.format(fake.norm().data[0], real.norm().data[0]))
            #utils.save_multiple_imgs(fake.data.cpu().numpy()[:config.NUM_SAVE], sample_path + '/epoch_{}_sample'.format(current_epoch))
            #utils.save_multiple_imgs(real.data.cpu().numpy()[:config.NUM_SAVE], sample_path + '/epoch_{}_real'.format(current_epoch))
            torchvision.utils.save_image(real.data[:config.NUM_SAVE], '{0}/epoch_{1}_real.png'.format(sample_path, current_epoch))
            torchvision.utils.save_image(fake.data[:config.NUM_SAVE], '{0}/epoch_{1}_fake.png'.format(sample_path, current_epoch))
            logger('vis: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))
            gen.train()
       
        if config.DECAY and (current_epoch % args.de == 0):
            current_dis_lr = current_dis_lr * 0.5
            current_gen_lr = current_gen_lr * 0.5
            for param_group in gen_optim.param_groups:
                param_group['lr'] = current_gen_lr
            for param_group in dis_optim.param_groups:
                param_group['lr'] = current_dis_lr
            logger('Decay learning rate to gen_lr: {0}, dis_lr: {1}.'.format(current_gen_lr, current_dis_lr))
        
        if gan_t == 'coop' and current_epoch % (config.MAX_EPOCH * config.ALPHA_STEP) == 0:
            alpha = float(current_epoch) / (config.MAX_EPOCH * config.ALPHA_STEP) * config.ALPHA_STEP
            logger('epoch: {0}/{1}, change alpha to: {2}'.format(current_epoch, config.MAX_EPOCH, alpha))

        current_epoch += 1

    logger('')
    logger('config:')
    with open('config.py', 'r') as fo:
        logger(fo.read())

if __name__ == '__main__':
    if args.cat == 'all':
        if args.data == 'cifar':
            cats = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        elif args.data == 'lsun':
            cats = ['classroom', 'church_outdoor']
    
        for c in cats:
            train(args.data, c)
    else:
        if args.data == 'cifar': 
            train(args.data, args.cat)
        elif args.data == 'lsun':
            train(args.data, args.cat)

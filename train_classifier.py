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


os.environ["CUDA_VISIBLE_DEVICES"] = args.dev if args.dev else str(1)
#------------------------------------------------------------#

#----------------


def train():
    config.MAX_EPOCH = args.epoch
    logger, path, sample_path, params_path = utils.exp_folder_init(args.exp, args.epoch, 'mixed', args.resume, '', 'lsun')
    logger('exp_folder: {}'.format(path))
    logger('res_folder: {}'.format(sample_path))
#------------------create components--------------------------#
    logger('init components')
    
    img_size = config.LSUN_IMG_SIZE
   
    resnet = torchvision.models.resnet18(pretrained=False)
    res_params = list(resnet.parameters())

    res_optim = optim.SGD(res_params, lr=config.GEN_LEARNING_RATE)

    criterion = nn.CrossEntropyLoss().cuda()

    n_class = 5
    resnet.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, n_class)

    
    resnet = resnet.cuda() 
    if args.resume:
        res_params = params_path + '/resnet_best.pkl'
        resnet.load_state_dict(torch.load(res_params))
        logger('load resnet from: {0}'.format(res_params))

    if config.FIXED_Z:
        fix_z = Variable(torch.randn(config.BATCH_SIZE, config.Z_SIZE)).cuda()

    logger('init complete')
    #------------------------------------------------------------#

    
    current_res_lr = config.GEN_LEARNING_RATE
       
    lsun_transform = transforms.Compose([transforms.Resize((config.LSUN_IMG_SIZE,config.LSUN_IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cat_train = ['bedroom_train', 'kitchen_train', 'restaurant_train', 'dining_room_train', 'living_room_train']
    cat_test = ['bedroom_val', 'kitchen_val', 'restaurant_val', 'dining_room_val', 'living_room_val']
    train_data = torchvision.datasets.LSUN(config.LSUN_PATH, cat_train, transform=lsun_transform)
    test_data = torchvision.datasets.LSUN(config.LSUN_PATH, cat_test, transform=lsun_transform)           

    train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = args.workers)
    test_loader = DataLoader(test_data, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = 1)

    best_epoch = 0
    current_epoch = 1
    best_prec = 0.5
    start_time = time.time()
    while current_epoch <= config.MAX_EPOCH:
          
        for (i, b) in enumerate(train_loader): 
            if i >= 10000:
                break
            real_data, target_label = b
            #update dis
            real = Variable(real_data.float())
            real = real.cuda()
            target_label = Variable(target_label)
            target_label = target_label.cuda()
            batch_size = real.size(0)

            resnet.zero_grad()
            logits = resnet(real)
            res_loss = criterion(logits, target_label)

            res_loss.backward()
            res_optim.step()
       
        if current_epoch % args.pe == 0:
            
            logger('epoch: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))
            logger('loss is  {:.2f}'.format(res_loss.data[0]))
            logger('time spent: {:.4f}'.format(time.time() - start_time))

            start_time = time.time()

            vst = time.time()

            resnet.eval()
            total_prec = 0
            correct = 0
            total = 0
            for (i, b) in enumerate(test_loader):
                input_data, target = b
                input_data = Variable(input_data.float()).cuda()
                output = resnet(input_data)
                _, prec = torch.max(output.data, 1)
                correct += (prec.cpu() == target).sum()
                total += input_data.size(0)                 

            total_prec =  float(correct) / total

            
            logger('acc is  {:.2f}'.format(total_prec))
            logger('time spent: {:.4f}'.format(time.time() - vst))

            logger('save model: {0}/{1}'.format(current_epoch, config.MAX_EPOCH))
            torch.save(resnet.state_dict(), os.path.join(params_path, 'resnet_{0}.pkl'.format(current_epoch)))
            if total_prec > best_prec:
                best_prec = total_prec
                best_epoch = current_epoch
                torch.save(resnet.state_dict(), os.path.join(params_path, 'resnet_best.pkl'))

            logger('best epoch: {}'.format(best_epoch))
            resnet.train()
       
        if config.DECAY and (current_epoch % args.de == 0):
            current_res_lr = current_res_lr * 0.5
        
            for param_group in res_optim.param_groups:
                param_group['lr'] = current_res_lr
    
            logger('Decay learning rate to lr: {0}'.format(current_res_lr))

        current_epoch += 1

    logger('')
    logger('config:')
    with open('config.py', 'r') as fo:
        logger(fo.read())

if __name__ == '__main__':
    train()
        

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import Dataset
import numpy as np
import cPickle
import config
import glob
import lmdb
import os
import pickle
import six
from PIL import Image
from PIL.Image import EXTENT


class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, cat):
        super(CIFAR10Dataset, self).__init__()

        #3072 = 3 * 32 * 32

        label_names = self.unpickle(data_path + '/batches.meta')['label_names']
        target_label = ''
        for (i,c) in enumerate(label_names):
            if c == cat:
                target_label = i
                break

        data_batch = glob.glob(data_path + 'data_batch_*')
        
        print('loading from:', data_batch)
        all_data = {}
        all_data['label'] = []
        all_data['data'] = []
        for i in data_batch:
            data_dict = self.unpickle(i)
            if all_data['data'] == []:
                all_data['data'] = np.array(data_dict['data'])
                all_data['label'] = np.array(data_dict['labels'])
            else: 
                all_data['data'] = np.concatenate((all_data['data'], data_dict['data']), 0)
                all_data['label'] = np.concatenate((all_data['label'], data_dict['labels']))
                    
        all_data['data'] = np.array(all_data['data'])

        if not cat == 'mixed':
            idx = np.array([i for i,c in enumerate(all_data['label']) if c == target_label])
            self.data = all_data['data'][idx]
        else:
            self.data = all_data['data']

        if config.G_LAST == 'sigmoid':
            '''sigmoid img ~ [0,1]
            '''
            self.data = self.data / 256.0
        elif config.G_LAST == 'tanh':
            '''tanh img ~ [-1,1]
            '''
            self.data = self.data - 128
            self.data = self.data / 128.0

        self.data = np.reshape(self.data, (-1,3,32,32)).astype('float')

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)

        return dict

    def __len__(self):
        return self.data.shape[0]
 
    def __getitem__(self, idx):
        return self.data[idx]

class LSUNDataset(Dataset):
    def __init__(self, root, cat):
        self.root = root + cat + '_train_lmdb/'
        self.default_size = config.LSUN_IMG_SIZE
        self.env = lmdb.open(self.root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + self.root.replace('/', '_')
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img = None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
    
        if img.size[0]>img.size[1]:
            offset=int(img.size[0]-img.size[1])/2
            img=img.transform((img.size[1],img.size[1]),EXTENT,(offset,0,int(img.size[0]-offset),img.size[1]))
        else:
            offset=int(img.size[1]-img.size[0])/2
            img=img.transform((img.size[0],img.size[0]),EXTENT,(0,offset,img.size[0],(img.size[1]-offset)))

        if img.size[0] != self.default_size:
            img.resize((self.default_size, self.default_size))

        img = np.array(img).transpose(2,0,1)
        if config.G_LAST == 'sigmoid':
            '''sigmoid img ~ [0,1]
            '''
            img = img / 256.0
        elif config.G_LAST == 'tanh':
            '''tanh img ~ [-1,1]
            '''
            img = img - 128
            img = img / 128.0

            img = img.astype('float')

        return img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'

class GeneratorCIFAR(nn.Module):
    def __init__(self, z_size):
        super(GeneratorCIFAR, self).__init__()
        self.base_channel = 32
        self.kernel_size = 4
        self.base_resolution = 2
        self.z_size = z_size       

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.z_size, out_channels=self.base_channel*8, kernel_size=self.kernel_size, stride=2, padding=1),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        
        self.layer2 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=self.base_channel*8, out_channels=self.base_channel*4, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*4),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*4, out_channels=self.base_channel*2, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*2),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*2, out_channels=self.base_channel, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel, out_channels=3, kernel_size=self.kernel_size, stride=2, padding=1),
        )

    def forward(self, z):

        z = z.view(-1, self.z_size, 1, 1)
        x1 = self.layer1(z)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.layer5(x4)       
    
        if config.G_LAST == 'tanh':
            act = nn.Tanh()
        elif config.G_LAST == 'sigmoid':
            act = nn.Sigmoid()
        
        return act(out)

class DiscriminatorCIFAR(nn.Module):
    def __init__(self):
        super(DiscriminatorCIFAR,self).__init__()
        self.base_channel = 32
        self.kernel_size = 4
        self.base_resolution = 2

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.base_channel, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel*2, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*2), 
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*2, out_channels=self.base_channel*4, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*4),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*4, out_channels=self.base_channel*8, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*8),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*8, out_channels=1, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.layer5(x4)
    
        return out.view(-1)
        
 
class GeneratorLSUN(nn.Module):
    def __init__(self, z_size):
        super(GeneratorLSUN, self).__init__()
        self.base_channel = 16
        self.kernel_size = 4
        self.base_resolution = 2
        self.z_size = z_size       

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.z_size, out_channels=self.base_channel*64, kernel_size=self.kernel_size, stride=2, padding=1),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        
        self.layer4 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=self.base_channel*64, out_channels=self.base_channel*32, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*32),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*32, out_channels=self.base_channel*16, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*16),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        self.layer16 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*16, out_channels=self.base_channel*8, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*8),
            #nn.ReLU()
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer32 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*8, out_channels=self.base_channel*4, kernel_size=self.kernel_size, stride=2, padding=1),
            #nn.ReLU()
            nn.BatchNorm2d(self.base_channel*4),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        
        self.layer64 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*4, out_channels=self.base_channel*2, kernel_size=self.kernel_size, stride=2, padding=1),
            #nn.ReLU()
            nn.BatchNorm2d(self.base_channel*2),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        
        self.layer128 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel*2, out_channels=self.base_channel, kernel_size=self.kernel_size, stride=2, padding=1),
            #nn.ReLU()
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )
        
        self.layer256 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.base_channel, out_channels=3, kernel_size=self.kernel_size, stride=2, padding=1),
        )

    def forward(self, z):

        z = z.view(-1, self.z_size, 1, 1)
        x2 = self.layer2(z)
        x4 = self.layer4(x2)
        x8 = self.layer8(x4)
        x16 = self.layer16(x8)
        x32 = self.layer32(x16)     
        x64 = self.layer64(x32)     
        x128 = self.layer128(x64)       
        out = self.layer256(x128)       
    
        if config.G_LAST == 'tanh':
            act = nn.Tanh()
        elif config.G_LAST == 'sigmoid':
            act = nn.Sigmoid()
        
        return act(out)

class DiscriminatorLSUN(nn.Module):
    def __init__(self):
        super(DiscriminatorLSUN,self).__init__()
        self.base_channel = 16
        self.kernel_size = 4
        self.base_resolution = 2

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.base_channel, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel*2, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*2), 
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*2, out_channels=self.base_channel*4, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*4),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*4, out_channels=self.base_channel*8, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*8),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*8, out_channels=self.base_channel*16, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*16),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*16, out_channels=self.base_channel*32, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*32),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*32, out_channels=self.base_channel*64, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channel*64),
            nn.LeakyReLU(config.LRELU_SLOPE)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel*64, out_channels=1, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)    
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        out = self.layer8(x7)

        return out.view(-1)
        

class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
           
        act = nn.Sigmoid()
        return act(output.view(-1))

class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        return output 

class MLP_G(nn.Module):
    def __init__(self, z_size, gaussian_size):
        super(MLP_G, self).__init__()

        main = nn.Sequential()
        main.add_module('l1', nn.Linear(z_size, 128))
        main.add_module('r1', nn.ReLU(True))
        main.add_module('l2', nn.Linear(128, 128))
        main.add_module('r2', nn.ReLU(True))
        main.add_module('l3', nn.Linear(128, gaussian_size))
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output 

class MLP_D(nn.Module):
    def __init__(self, gaussian_size):
        super(MLP_D, self).__init__()
        main = nn.Sequential()
        main.add_module('l1', nn.Linear(gaussian_size, 128))
        main.add_module('lr1', nn.LeakyReLU(0.2, inplace=True))
        main.add_module('l2', nn.Linear(128, 128))
        main.add_module('lr2', nn.LeakyReLU(0.2, inplace=True))
        main.add_module('l3', nn.Linear(128, 1))
        main.add_module('sigmoid', nn.Sigmoid())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

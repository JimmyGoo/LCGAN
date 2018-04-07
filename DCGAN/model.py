import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle

class CIFAR100Dataset(Dataset):
    def __init__(self, data_path):
        super(CIFAR100Dataset, self).__init__()

        #3072 = 3 * 32 * 32

        self.data_dict = self.unpickle(data_path)
        self.data = self.data_dict[b'data']
        self.data = np.reshape(self.data, (-1,3,32,32)).astype('float')
        self.data = np.transpose(self.data, (0,2,3,1))

    def unpickle(self, file):
        
        print('cifar-100 unpickle')

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        return dict

    def __len__(self):
        return self.data.shape[0]
 
    def __getitem__(self, idx):
        return self.data[idx]


class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.base_channel = 32
        self.kernel_size = 4

        self.fc = nn.Linear(z_size, self.base_channel)

        self.deconv1 = nn.ConvTranspose2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.deconv1_bn = nn.BatchNorm2d(self.base_channel)

        self.deconv2 = nn.ConvTranspose2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.deconv2_bn = nn.BatchNorm2d(self.base_channel)

        self.deconv3 = nn.ConvTranspose2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.deconv3_bn = nn.BatchNorm2d(self.base_channel)

        self.deconv4 = nn.ConvTranspose2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.deconv4_bn = nn.BatchNorm2d(self.base_channel)


    def forward(self, z):

        x1 = F.relu(self.deconv1_bn(self.deconv1(z))) 
        x2 = F.relu(self.deconv2_bn(self.deconv2(x1))) 
        x3 = F.relu(self.deconv3_bn(self.deconv3(x2))) 
        x4 = self.deconv4_bn(self.deconv4(x3))
        out = F.tanh(x4)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.base_channel = 32
        self.kernel_size = 4

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.conv1_bn = nn.BatchNorm2d(self.base_channel)

        self.conv2 = nn.Conv2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.conv2_bn = nn.BatchNorm2d(self.base_channel)

        self.conv3 = nn.Conv2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.conv3_bn = nn.BatchNorm2d(self.base_channel)

        self.conv4 = nn.Conv2d(in_channels = self.base_channel, out_channels = self.base_channel, 
            kernel_size = self.kernel_size, stride = 2, padding = 0)
        self.conv4_bn = nn.BatchNorm2d(self.base_channel)

        self.fc = nn.Linear(self.base_channel, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1_bn(self.conv1(x)), negative_slope = config.LRELU_SLOPE)
        x2 = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope = config.LRELU_SLOPE)
        x3 = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope = config.LRELU_SLOPE)
        x4 = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope = config.LRELU_SLOPE)
        out = F.leaky_relu(self.fc(x4))

        return out
        

 
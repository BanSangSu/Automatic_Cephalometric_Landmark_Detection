import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from mytransforms import *
from mytransforms import mytransforms
from skimage.filters import threshold_otsu
from skimage import feature
from skimage.color import rgb2gray
from numpy import matlib
import cv2
import os,sys
import numpy as np
from numpy import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

def gray_to_rgb(gray):
    h,w = gray.shape
    rgb=np.zeros((h,w,3))
    rgb[:,:,0]=gray;    rgb[:,:,1]=gray;    rgb[:,:,2]=gray;
    return rgb

class dataload(Dataset):
    def __init__(self,  path='train', H=600,W=480,pow_n=3, aug=True):

        init_trans = transforms.Compose([transforms.Resize((H, W)),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])
        self.datainfo = torchvision.datasets.ImageFolder(root=path, transform=init_trans)
        self.mask_num=len(self.datainfo.classes)-1
        self.data_num = int(len(self.datainfo)/len(self.datainfo.classes))
        self.aug=aug
        self.pow_n = pow_n
        self.task = path
        self.H = H
        self.W = W


    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        if self.aug == True: self.rv = random.random()
        else: self.rv=-1
        if self.rv>=.1:
            # augmenation of img and masks
            # angle = random.randrange(-25, 25)
            angle = random.randrange(-50, 50) #changeBan
            trans_rand = [random.uniform(0, 0.05) , random.uniform(0, 0.05)]
            scale_rand = random.uniform(0.9, 1.1)
            # trans img with masks
            self.input_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                     mytransforms.Affine(angle,
                                                                         translate=trans_rand,
                                                                         scale=scale_rand,
                                                                         fillcolor=0),
                                                     mytransforms.ToTensor(),
                                                     ])
            self.mask_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                    mytransforms.Affine(angle,
                                                                        translate=trans_rand,
                                                                        scale=scale_rand,
                                                                        fillcolor=0),
                                                    mytransforms.ToTensor(),
                                                    ])

            self.col_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                   mytransforms.ColorJitter(brightness=random.random(),
                                                                            contrast=random.random(),
                                                                            saturation=random.random(),
                                                                            hue=random.random() / 2
                                                                            ),
                                                   mytransforms.ToTensor(),
                                                   ])

            #print("angle:", angle, "vfilp:", vfilp)
            image, _ = self.datainfo.__getitem__(idx)

            #plt.imshow(image[0], cmap='gray');plt.show()
            image = self.col_trans(image)
            image = self.input_trans(image)

            #plt.imshow(image[0], cmap= 'gray' ); plt.show()
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)

            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = self.mask_trans(X)
####################################################
        else:
            image, _ = self.datainfo.__getitem__(idx)
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = X

        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()
        #print("idx :", idx, "path ", self.task)

        #plt.imshow(image[0], cmap='gray');   plt.show()
        return [image, mask ]

#changeBan
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
)


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

  #changeBan
        self.base_model = torchvision.models.resnet18(pretrained=True)

        self.base_layers = list(self.base_model.children())

        #gray
        self.base_layers[0] = nn.Conv2d(img_ch, 64,kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        self.layer0 = nn.Sequential(*self.base_layers[0:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(img_ch, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, output_ch, 1)
        
    def forward(self, input):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


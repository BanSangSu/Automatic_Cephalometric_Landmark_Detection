import os,sys
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from collections import defaultdict
import torch.nn.functional as F
import cv2 as cv
from numpy import *
from scipy.spatial import distance

device_txt = 'cuda:0'
device = torch.device( device_txt if torch.cuda.is_available() else "cpu")

num_class = 10

### parameter modification ##########
H=800; W=640;
from net_Unet_Uetpp import *
import net_Unet_Uetpp as network

model = network.U_Net(img_ch=1, output_ch=num_class).to(device);
model.load_state_dict(torch.load('model(400,+-10,3e-4,10)/Network_0.000319618220673874_E_359.pth',map_location=device_txt)) # model loading
model=model.eval()

data = dataload(path='data/val', H=H, W=W, aug=False);

# Data loading with preprocessing
x = data.__getitem__(15)
inputs = x[0]
inputs = inputs.unsqueeze(0)
label = x[1]
label = label.unsqueeze(0)
inputs = inputs.to(device)
outputs = model(inputs.data)
output = outputs.data

#########################################
def gray_to_rgb(gray):
    h,w = gray.shape
    rgb=np.zeros((h,w,3))
    rgb[:,:,0]=gray;    rgb[:,:,1]=gray;    rgb[:,:,2]=gray;
    return rgb

mtx = gray_to_rgb(x[0][0])
pred_mtx =[]; gt_mtx =[];
for k in range(0, num_class):
    A = outputs[0][k]; A=A.cpu()
    B = label[0][k]; B=B.cpu()

    pred = np.array(np.where(A > A.max() * .95)); pred = pred.mean(axis=1)
    GT = np.array(np.where(B > B.max() * .95));   GT = GT.mean(axis=1)

    GT=np.round(GT)
    cv.circle(mtx, (int(GT[1]), int(GT[0])), 2, (0,1,0), 3); # GT
    pred=np.round(pred)
    cv.circle(mtx, (int(pred[1]), int(pred[0])), 2, (0, 0, 1), 3);  # pred

    print("GT:" ,GT , "Pred:", pred)
    pred_mtx.append(pred)
    gt_mtx.append(GT)


plt.imshow(mtx);plt.show()

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

### parameter modification ##########
from net import *
import net as network

num_class = 10
H=800; W=640;

model = network.U_Netpp(img_ch=1, output_ch=num_class).to(device);
model.load_state_dict(torch.load('model/E300.pth',map_location=device_txt)) # model loading
model=model.eval()

# Data loading with preprocessing
preprocess = transforms.Compose([transforms.Resize((H, W)),
                                 transforms.Grayscale(1),
                                 transforms.ToTensor(),
                                 ])
datainfo = torchvision.datasets.ImageFolder(root='data/test', transform=preprocess)

## predict landmarks
subjects=[]
for k in range(0,10):
    x = datainfo.__getitem__(k)
    x = x[0].unsqueeze(0).to(device)
    outputs = model(x)

    subject = []
    for land in range(0, 10):
        A = outputs[0][land];
        A = A.cpu()
        pred = np.array(np.where(A > A.max() * .95));
        pred = pred.mean(axis=1);
        pred = np.round(pred)
        subject.append(pred)
    subjects.append(subject)

subjects=np.array(subjects)
np.save('ㅌㅌㅌ.npy', subjects)

## 시각화
k=5
x = datainfo.__getitem__(k)
x = x[0].unsqueeze(0).to(device)
outputs = model(x)
mtx = gray_to_rgb(x[0][0].cpu())
pred_mtx =[];
pred=subjects[k]
print("Pred:", pred)
for k in range(0, num_class):
    cv.circle(mtx, (int(pred[k][1]), int(pred[k][0])), 2, (0, 0, 1), 3);  # pred
plt.imshow(mtx);plt.show()



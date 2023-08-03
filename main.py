import os,sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
from net_Unet_Uetpp import *
import net_Unet_Uetpp as network

#os.chdir("/content/drive/MyDrive/ISBI_pytorch/ISBI_pytorch")
batch_size = 1
H=800; W=640;
# dataloaders = {
#     'train': DataLoader(dataload(path='data/train', H=H, W=W,pow_n=8, aug=True) , batch_size=batch_size, shuffle=True, num_workers=3),
#     'val': DataLoader(dataload(path='data/val', H=H, W=W, pow_n=8, aug=False), batch_size=batch_size, shuffle=False, num_workers=3)
# }
dataloaders = {#changeBan
    'train': DataLoader(dataload(path='data/train', H=H, W=W,pow_n=2, aug=True) , batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(dataload(path='data/val', H=H, W=W, pow_n=2, aug=False), batch_size=batch_size, shuffle=False, num_workers=4)
}

from collections import defaultdict

# def L1_loss(pred, target):
#     loss = torch.mean(torch.abs(pred - target))
#     metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
#     return loss
# def L2_loss(pred, target):
#     loss = torch.mean(torch.pow((pred - target), 2))
#     metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
#     return loss
criterion = nn.MSELoss() #changeBan

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    #model=torch.load('BEST.pt').to(device)
    # model=network.U_Net(img_ch=1, output_ch=10).to(device)
    #model.load_state_dict(torch.load('model/newL1_0.00015339434321504086_E_69.pth',map_location=device_txt))

    # model= network.U_Net(img_ch=1,output_ch=10).to(device)
    model= network.U_Net(1,10).to(device)#changeBan
   # model = network.DeepLabv3_plus(nInputChannels=1, n_classes=10, os=16, pretrained=False, _print=True)
   # model.to(device)
    # Observe that all parameters are being optimized
    num_epochs = 1000
    #optimizer = optim.Adam(model.parameters(), lr=3e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # optimizer = adabound.AdaBound(model.parameters(), lr=1e-4)#changBan
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    # scheduler = custom_scheduler.CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=1e-1,  T_up=10, gamma=1)#changeBan

    print("****************************GPU : ", device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    valtest = 10
    for epoch in range(num_epochs):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('------------------------' * 10)
        now = time.time()

        if (epoch + 1) % valtest == 0:
            uu = ['train', 'val']
        else:
            uu = ['train']

        for phase in uu:
            # since = time.time()
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float) # 성능 값 중첩
            epoch_samples = 0

            num_ = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                num_ = num_ + 1
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation
                    outputs = model(inputs) #clsout : 8by3  // cls_label : 1 by 8
                    # acloss = L2_loss(outputs, labels)
                    # loss=acloss
                    loss = criterion(outputs, labels) #changeBan
                    # metrics['Jointloss'] += loss
                    metrics['Jointloss'] += loss.item() #changeBan
                    #print("loss :" , loss, "integ loss", metrics['Jointloss'])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                # statistics
                epoch_samples += inputs.size(0)

            # print_metrics(metrics, epoch_samples, phase)

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            print(phase,"Joint loss :", epoch_Jointloss )

            # deep copy the model

            savepath = 'model/Network_{}_E_{}.pth'
            if phase == 'val' and epoch_Jointloss < best_loss:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))

            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_loss = epoch_Jointloss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))

        print(time.time() - now)

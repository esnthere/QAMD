import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial
import matplotlib.pyplot as plt
import lmdb
from prefetch_generator import BackgroundGenerator
from torch.cuda.amp import autocast as autocast
import copy
from DISTS_pt import DISTS
from lpips import lpips


class Mydataset(Dataset):
    def __init__(self, nms,datapath,tgtpath):
        self.nms = nms
        self.datapath = datapath
        self.tgtpath = tgtpath
    def __getitem__(self, index):
        x=cv2.cvtColor(cv2.imread(self.datapath+'/' + self.nms[index], 1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        y=cv2.cvtColor(cv2.imread(self.tgtpath+'/' + self.nms[index], 1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return torch.from_numpy(x),torch.from_numpy(y),self.nms[index]

    def __len__(self):
        return len(self.nms)


def test(model_loss,model_loss2, test_loader, epoch, device, all_test_loss):
    model_loss.eval()
    model_loss2.eval()

    all_loss4 = 0
    all_loss4_2 = 0
    all_data = 0
    lp=[]
    dt=[]
    nms=[]
    for batch_idx, (data, target,nm) in enumerate(test_loader):
        data, target, = data.to(device), target.to(device)
        data = ((data.float()) / 255.0)
        target = ((target.float()) / 255.0)
        with torch.no_grad():
            with autocast():
                lpt=model_loss2.forward(data * 2 - 1, target * 2 - 1)
                lp+=list(lpt[:,0,0,0].cpu().numpy())
                loss4_2 = torch.mean(lpt)
                output_large_r = F.interpolate(data, size=(256, 256), mode='area').clamp(min=0, max=1)
                target_large_r = F.interpolate(target, size=(256, 256), mode='area').clamp(min=0, max=1)
                # print(target.shape)
                dtt=model_loss((output_large_r), (target_large_r))
                dt+=list(dtt.cpu().numpy())
                loss4 = torch.mean(dtt)
                nms+=list(nm)

        all_loss4 += loss4.item() * data.shape[0]
        all_loss4_2 += loss4_2.item() * data.shape[0]

        all_data += data.shape[0]

    print('Test Epoch:{} \t Mean Loss4: {:.4f} Loss4_2: {:.4f}'.format( epoch, all_loss4 / all_data, all_loss4_2 / all_data))



    return dt,lp,nms


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")

    model_loss = DISTS()
    # model_loss = BalancedDataParallel(num // 1, model_loss, dim=0)
    model_loss = nn.DataParallel(model_loss).to(device)

    model_loss2 = lpips.LPIPS(net='alex')
    # model_loss = BalancedDataParallel(num // 1, model_loss, dim=0)
    model_loss2 = nn.DataParallel(model_loss2).to(device)

    methods = [ 'Restormer']
    datasets = [ 'Hide']

    for j in range(len(datasets)):
        all_dists = []
        all_lpips = []
        all_nms = []
        for i in range(len(methods)):
            print (methods[i], ' on ', datasets[j], ':')
            datapath = ".\\results\\" + datasets[j]
            tgtpath = "F:\STS\Deblur\Restormer\gt_results\\" + datasets[j]
            nms=os.listdir(tgtpath)
            test_dataset = Mydataset(nms,datapath,tgtpath)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16,pin_memory=True)
            start = time.time()
            dt,lp,nm=test(model_loss, model_loss2,test_loader, -1, device, [])
            all_dists.append( dt)
            all_lpips.append( lp)
            all_nms.append( nm)
            print("time:", time.time() - start)
        savenm=datasets[j]+'_cross_scores.mat'
        sio.savemat(savenm, {'dists': all_dists, 'lpips': all_lpips, 'nms': all_nms})


if __name__ == '__main__':
    main()


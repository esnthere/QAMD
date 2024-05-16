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
from myrestormer_arch import  Restormer
from myrestormer_arch2 import  Restormer as PredNet

from efficientnet_3fc_norelu_drp05 import EfficientNet

from BalancedDataParallel import BalancedDataParallel
from DISTS_pt import DISTS


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs,targets, imgs_large,targets_large):
        self.imgs = imgs
        self.targets = targets

        self.imgs_large = imgs_large
        self.targets_large = targets_large

    def __getitem__(self, index):
        x=cv2.cvtColor(cv2.imread(imgpath+'/' + self.imgs_large[index], 1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        st1=int((x.shape[1]-736)/2)
        st2=int((x.shape[2]-640)/2)
        x=x[:,st1:st1+736,st2:st2+640]
        y=cv2.cvtColor(cv2.imread(imgpath+'/' + self.targets_large[index], 1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        st1 = int((y.shape[1] - 736) / 2)
        st2 = int((y.shape[2] - 640) / 2)
        y = y[:, st1:st1 + 736, st2:st2 + 640]
        return torch.from_numpy(self.imgs[index]),torch.from_numpy(self.targets[index]),torch.from_numpy(x),torch.from_numpy(y)

    def __len__(self):
        return (self.imgs).shape[0]

def show_output(data,op_ims,target):

    target = np.transpose((target+1)/2, (0, 2, 3, 1))
    op_ims = np.transpose((op_ims+1)/2,  (0, 2, 3, 1))
    data = np.transpose((data+1)/2,  (0, 2, 3, 1))
    for n,i in enumerate (np.linspace(0, data.shape[0]-1, 9).astype(int)):
        if n==0:
            im=np.concatenate((data[i],op_ims[i],target[i]),axis=1)
        else:
            im=np.concatenate((im,np.concatenate((data[i],op_ims[i],target[i]),axis=1)),axis=0)

    plt.figure(figsize=(7*9, 7*4))
    plt.imshow(im)
    plt.axis('off')
    plt.show()

    return 0


def re_norm(x):
    x=(x+1)/2
    return x

def myrenorm(target):
    targetn=(target+0)/1
    targetn[:, 0] -= 0.485
    targetn[:, 1] -= 0.456
    targetn[:, 2] -= 0.406
    targetn[:, 0] /= 0.229
    targetn[:, 1] /= 0.224
    targetn[:, 2] /= 0.225
    return targetn

def myaug(inp_img,tar_img):
    ps=128

    hh, ww = tar_img.shape[2], tar_img.shape[3]

    rr = np.random.randint(0, hh - ps)
    cc = np.random.randint(0, ww - ps)

    aug = np.random.randint(0, 3)

    # Crop patch
    inp_img = inp_img[:,:, rr:rr + ps, cc:cc + ps]
    tar_img = tar_img[:,:, rr:rr + ps, cc:cc + ps]

    # Data Augmentations
    if aug == 1:
        inp_img = torchvision.transforms.functional.hflip(inp_img)
        tar_img = torchvision.transforms.functional.hflip(tar_img)
    elif aug == 2:
        inp_img = torchvision.transforms.functional.vflip(inp_img)
        tar_img = torchvision.transforms.functional.vflip(tar_img)


    return   inp_img,tar_img

def train(model,model_pred, model_loss, model_IQA,ms,train_loader, optimizer, scaler, epoch, device, all_train_loss):
    model.train()
    model_loss.eval()

    all_loss = 0
    all_loss1 = 0
    all_loss2 = 0
    all_loss3_1 = 0
    all_loss3_2 = 0
    all_loss3_3 = 0
    all_data = 0

    for batch_idx, (data,  target,  data_large, target_large) in enumerate(train_loader):

        model_dict = model_pred.state_dict()
        pretrained_dict = model.state_dict()
        for k, v in model_dict.items():
            model_dict[k] = pretrained_dict[k]
        model_pred.load_state_dict(model_dict)
        model_pred.eval()

        data,  target, = data.to(device),  target.to(device)
        data_large,target_large = data_large.to(device),  target_large.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(128+24, (3,))
        data = data[:, :, rd_ps[0] * 1:rd_ps[0] * 1 + 128, rd_ps[1] * 1:rd_ps[1] * 1 + 128]
        target = target[:, :, rd_ps[0] * 1:rd_ps[0] * 1 + 128, rd_ps[1] * 1:rd_ps[1] * 1 + 128]
        if rd_ps[1] < 12+64:
            data = torch.flip(data, dims=[3])
            target = torch.flip(target, dims=[3])

        data = ((data.float()) / 255.0)
        target = ((target.float()) / 255.0)

        data_large, target_large=myaug(data_large,target_large)
        data_large = ((data_large.float()) / 255.0)
        target_large = ((target_large.float()) / 255.0)
        with autocast():
            with torch.no_grad():
                tg = model_IQA(myrenorm(target)) / ms[0, 6]
                ft = model_IQA(myrenorm(data)) / ms[0, 6]

        with autocast():
            optimizer.zero_grad()
            output_large = model(data_large)

            if batch_idx%2 ==1:
                output = model(data)
                ft_db = model_pred(output[0])
                loss3_1 = F.mse_loss(output[1] / ms[0, 6], ft)
                loss3_2 = F.mse_loss(ft_db / ms[0, 6], tg)
            else:
                output = model(target)
                loss3_1 = F.mse_loss(output[1] / ms[0, 6], tg)


            loss1 = F.l1_loss(output_large[0], target_large)
            loss2 = F.l1_loss(output[0], target)

            if batch_idx%2 ==1:
                loss = loss1+ 0.5*loss2 + 0.5*loss3_1 + 0.5*loss3_2
                all_loss3_2 += loss3_2.item() * data.shape[0]
            else:
                loss = loss1+ 0.5*loss2 + 0.5*loss3_1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            all_loss += loss.item() * data.shape[0]
            all_loss1 += loss1.item() * data.shape[0]
            all_loss2  += loss2.item()*data.shape[0]
            all_loss3_1 += loss3_1.item() * data.shape[0]

            # all_loss3_3 += loss3_3.item() * data.shape[0]

            all_data += data.shape[0]

    print( 'Train Epoch:{} \t Mean Loss: {:.4f} Loss1: {:.4f}Loss2: {:.4f} Loss3_1: {:.4f}  Loss3_2: {:.4f} '.format(
            epoch, all_loss / all_data, all_loss1 / all_data,  all_loss2/ all_data,  all_loss3_1 / all_data, all_loss3_2/ all_data))
    return all_train_loss, all_loss / all_data


def test(model, model_loss, test_loader, epoch, device, all_test_loss):
    model.eval()
    model_loss.eval()

    all_loss = 0
    all_loss1 = 0
    all_loss2 = 0
    all_loss3_1 = 0
    all_loss3_2 = 0

    all_loss4 = 0
    all_loss4_2 = 0
    all_data = 0


    for batch_idx, (data, target, data_large, target_large) in enumerate(test_loader):
        data, target, = data.to(device), target.to(device)
        data_large, target_large = data_large.to(device), target_large.to(device)
        model_loss.eval()

        data = ((data.float()) / 255.0)
        target = ((target.float()) / 255.0)

        data_large = ((data_large.float()) / 255.0)
        target_large = ((target_large.float()) / 255.0)

        with torch.no_grad():
            with autocast():
                output_large = model(data_large)
                output_large_r = F.interpolate(output_large[0], size=(256, 256), mode='area').clamp(min=0, max=1)
                target_large_r = F.interpolate(target_large, size=(256, 256), mode='area').clamp(min=0, max=1)
                # print(target.shape)

                loss4 = torch.mean(model_loss((output_large_r), (target_large_r)))


        all_loss4 += loss4.item() * data.shape[0]

        all_data += data.shape[0]

    print('Test Epoch:{} \t Mean Loss4: {:.4f}'.format( epoch, all_loss4 / all_data))



    return all_test_loss, all_loss4 / all_data


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    device = torch.device("cuda")


    all_data = sio.loadmat('E:\Database\Deblur_datasets\RealBlur\Realblur_J_Area_280_140_70.mat')
    X = all_data['X']
    Y = all_data['Y']


    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']

    del all_data

    fname = 'E:\Database\Deblur_datasets\RealBlur\RealBlur_J_train_list.txt'
    X_large = []
    Y_large = []
    Xtest_large = []
    Ytest_large = []

    with open(fname, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            Y_large.append(i.split(' ')[0])
            X_large.append(i.split(' ')[1])

    fname = 'E:\Database\Deblur_datasets\RealBlur\RealBlur_J_test_list.txt'
    nms_test = []
    with open(fname, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            Ytest_large.append(i.split(' ')[0])
            Xtest_large.append(i.split(' ')[1])

    global  imgpath
    imgpath='E:\Database\Deblur_datasets\RealBlur'

    test_batch_size=4
    batch_size =8
    num=int(3)
    net=Restormer()
    model_dict = net.state_dict()
    pretrained_dict = torch.load('restormer_realj_dists_iqaft_bl.pt')
    for k, v in pretrained_dict.items():
        model_dict[k[7:]] = pretrained_dict[ k]
    net.load_state_dict(model_dict)
    # model=BalancedDataParallel(num // 1, net, dim=0).to(device)
    model=nn.DataParallel(net).to(device)

    net=PredNet( )
    # model_pred=BalancedDataParallel(num // 1, net, dim=0).to(device)
    model_pred=nn.DataParallel(net).to(device)

    model_loss= DISTS()
    # model_loss = BalancedDataParallel(num // 1, model_loss, dim=0)
    model_loss = nn.DataParallel(model_loss).to(device)

    model_IQA = EfficientNet.from_name('efficientnet-b0').to(device)
    model_dict = model_IQA.state_dict()  # 读出搭建的网络的参数，以便后边更新之后初始化
    pretrained_dict = torch.load('Koniq_efctb0_80_4.pt')
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict['module.' + k]
    model_IQA.load_state_dict(model_dict)
    model_IQA = BalancedDataParallel(num // 1, model_IQA, dim=0).to(device)
    ms = sio.loadmat('Gopro_effcientnet_mean_std.mat')['mean_and_std']


    for param in model_IQA.parameters():
        param.requires_grad = False
    for param in model_loss.parameters():
        param.requires_grad = False
    for param in model_pred.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    for param in model.module.pred.parameters():
        param.requires_grad = True

    # model.load_state_dict(torch.load( 'srnet_gopro.pt'))
    ###################################################################

    train_dataset = Mydataset(X,Y,X_large, Y_large)
    test_dataset = Mydataset(Xtest,Ytest,Xtest_large,  Ytest_large)

    epochs = 3000
    num_workers_train = 0
    num_workers_test = 0

    train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,pin_memory=True)
    test_loader = DataLoaderX(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=num_workers_test,pin_memory=True)

    all_train_loss = []
    all_test_loss = []

    start = time.time()
    # all_test_loss, _ = test(model,model_loss, test_loader, -1, device, all_test_loss)
    print("time:", time.time() - start)

    lr = 0.0001
    min_plsp = 200
    scaler =  torch.cuda.amp.GradScaler()
    global warm_epoch
    warm_epoch=3

    lr = 1e-5
    epochs=100
    ct = 0
    for epoch in range(epochs):
        print('lr:',lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)


        start = time.time()
        all_train_loss,train_loss = train(model,model_pred, model_loss,model_IQA,ms, train_loader, optimizer, scaler,  epoch, device, all_train_loss)
        print("time:", time.time() - start)
        if epoch==1:
            for nm, param in model.named_parameters():
                param.requires_grad = True
            ct = 0

        if epoch %2==0 and epoch>1:
            all_test_loss, loss = test(model, model_loss, test_loader, epoch, device, all_test_loss)
            print("time:", time.time() - start)
            ct+=1
            if min_plsp > loss:
                save_nm = 'restormer_realj_iqa.pt'
                min_plsp = loss
                torch.save(model.state_dict(), save_nm)
                ct=0

        if ct > 2:
            model.load_state_dict(torch.load(save_nm))
            lr *= 0.3
            ct = 0
            if lr < 1e-6:
                print("Train End! The best model:")
                model.load_state_dict(torch.load(save_nm))
                test(model,model_loss,  test_loader, epoch, device, all_test_loss)
                break


if __name__ == '__main__':
    main()


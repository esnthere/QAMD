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
import cv2
from torch.cuda.amp import autocast as autocast
import copy
from myrestormer_arch import  Restormer


from DISTS_pt import DISTS
from lpips import lpips

class Mydataset(Dataset):
    def __init__(self, imgs_large,targets_large,nms):

        self.imgs_large = imgs_large
        self.targets_large = targets_large
        self.nms = nms

    def __getitem__(self, index):
        x=cv2.cvtColor(cv2.imread(imgpath+'/' + self.imgs_large[index], 1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        y=cv2.cvtColor(cv2.imread(imgpath+'/' + self.targets_large[index], 1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return torch.from_numpy(x),torch.from_numpy(y),self.nms[index]

    def __len__(self):
        return len(self.nms)


def save_imgs(imgs,nms):
    for i in range(imgs.shape[0]):
        cv2.imwrite(savedir+'/'+nms[i].split('/')[1]+'_'+nms[i].split('/')[-1],cv2.cvtColor(imgs[i].transpose(1,2,0)*255, cv2.COLOR_RGB2BGR))


def test(model, model_loss,model_loss2, test_loader, epoch, device, all_test_loss):
    model.eval()
    model_loss.eval()

    all_loss4 = 0
    all_loss4_2 = 0
    all_data = 0


    for batch_idx, (data_large, target_large,nms) in enumerate(test_loader):

        data_large, target_large = data_large.to(device), target_large.to(device)
        model_loss.eval()

        data_large = ((data_large.float()) / 255.0)
        target_large = ((target_large.float()) / 255.0)

        with torch.no_grad():
            with autocast():
                output_large = model(data_large)
                save_imgs(output_large[0].cpu().numpy(), nms)
                loss4_2 = torch.mean(model_loss2.forward((output_large[0]), (target_large)))

                output_large_r = F.interpolate(output_large[0], size=(256, 256), mode='area').clamp(min=0, max=1)
                target_large_r = F.interpolate(target_large, size=(256, 256), mode='area').clamp(min=0, max=1)
                # print(target.shape)

                loss4 = torch.mean(model_loss((output_large_r), (target_large_r)))


        all_loss4 += loss4.item() * data_large.shape[0]
        all_loss4_2 += loss4_2.item() * data_large.shape[0]

        all_data += data_large.shape[0]

    print('Test Epoch:{} \t Mean Loss4: {:.4f} Loss4_2: {:.4f}'.format( epoch, all_loss4 / all_data, all_loss4_2 / all_data))

    return all_test_loss, all_loss4 / all_data


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")
    print('Hide:')
    global savedir
    savedir='./results\Hide'


    Xtest_large = []
    Ytest_large = []

    fname = 'E:\Database\Deblur_datasets\HIDE_dataset/test/far.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            Ytest_large.append('GT/' + i[:-1])
            Xtest_large.append('test/test-long-shot/' + i[:-1])

    fname = 'E:\Database\Deblur_datasets\HIDE_dataset/test/near.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            Ytest_large.append('GT/' + i[:-1])
            Xtest_large.append('test/test-close-ups/' + i[:-1])

    global imgpath
    imgpath = 'E:\Database\Deblur_datasets\HIDE_dataset'

    test_batch_size=1
    net=Restormer()
    model=nn.DataParallel(net).to(device)



    model_loss= DISTS()
    model_loss = nn.DataParallel(model_loss).to(device)

    model_loss2 =  lpips.LPIPS(net='alex')
    model_loss2 =nn.DataParallel(model_loss2).to(device)

    model.load_state_dict(torch.load( 'restormer_gopro_iqa.pt'))
    ###################################################################

    test_dataset = Mydataset(Xtest_large,  Ytest_large,Xtest_large)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=0,pin_memory=True)

    start = time.time()
    test(model,model_loss, model_loss2,test_loader, -1, device, [])
    print("time:", time.time() - start)


if __name__ == '__main__':
    main()



# QAMD:Quality-Aware Blind Image Motion Deblurring

This is the training example of QAMD on the GOPRO and RealBlur-J dataset with the baseline archtecture of Restormer. The trainning process is the same for other datasets:




## Dependencies and Installation
Pytorch: 2.1.2

CUDA: 12.1

Python: 3.11


## For training:

## 1. Data preparation

   To ensure high speed, save training images and lables into 'mat' files. Please run '**data_preparation.py**' to save the training images and labels, and '**Gopro_effcientnet_mean_std.mat**' contains the mean and standard deviation values for feature normalization.
   
## 2. Training the model

   Run'**restormer_Gopro_dists_iqaft.py**' to train the model. The model is pre-trained with patchsize of 128*128  on our device (Nidia TITANXP) as baseline, which can be obtained from: [Pre-trained models](https://pan.baidu.com/s/1sOqj_LGvtsHIN1pkCTh1yg?pwd=090d). Please download the file and put them in the same folder of code, The model in '**myrestormer_arch.py**' and '**myrestormer_arch2.py**' are modified from open accessed source code of [Restormer](https://github.com/swz30/Restormer). 
   
   The IQA model is modified from open accessed source code of [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch), and the pre-trained model can be obtained at [Pre-trained models](https://pan.baidu.com/s/1sOqj_LGvtsHIN1pkCTh1yg?pwd=090d).
   
   The files in the folder of '**lpips'** are obtained from open accessed source code of [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and the '**DISTS_pt.py**' is modified from  open accessed source code of [DISTS](https://github.com/dingkeyan93/DISTS) . 

   The training example can be seen from '**run restormer_Gopro_dists_iqaft.ipynb**' and  '**run restormer_RealJ_dists_iqaft.ipynb**'.

## If you like this work, please cite:

{

  author={Tianshu Song, Leida Li, Jinjian Wu, Weisheng Dong, Deqiang Cheng,},
  
  journal={Pattern Recognition}, 
  
  title={Quality-aware blind image motion deblurring}, 
  
  volume = {153},
 
  pages = {110568},

  year = {2024},

  doi = {https://doi.org/10.1016/j.patcog.2024.110568},
  
}

  
## License
This repository is released under the Apache 2.0 license. 



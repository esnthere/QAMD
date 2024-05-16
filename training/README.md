# QAMD:Quality-Aware Blind Image Motion Deblurring


## Dependencies and Installation
Pytorch: 2.1.2

CUDA: 12.1

Python: 3.11


## For training:

## 1. Data preparation

   To ensure high speed, save training images and lables into 'mat' files. Please run '**data_preparation.py**' to save the training images and labels, and '**Gopro_effcientnet_mean_std.mat**' contains the mean and standard deviation values for feature normalization.
   
## 2. Training the model

   Run'**restormer_Gopro_dists_iqaft.py**' to train the model. The model is modified  from open accessed source code of [Restormer](https://github.com/swz30/Restormer) and is pre-trained with patchsize of 128*128  on our device (Nidia TITANXP) as baseline, which can be obtained from: [Pre-trained models](https://pan.baidu.com/s/1YqQZaTRKPEdqYH3x6F4oJg?pwd=lv40 ). Please download the file and put them in the same folder of code, The model in '**myrestormer_arch.py**' and '**myrestormer_arch2.py**' are also modified from open accessed source code of [Restormer](https://github.com/swz30/Restormer). 
   
   The IQA model '**efficientnet_3fc_norelu_drp05.py**' is modified from open accessed source code of [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch), and the pre-trained model can be obtained at [Pre-trained models](https://pan.baidu.com/s/1YqQZaTRKPEdqYH3x6F4oJg?pwd=lv40 ).
   
   The files of '**DISTS_pt.py**' is modified from  open accessed source code of [DISTS](https://github.com/dingkeyan93/DISTS) and '**weights.pt**' contains the pre-trained weight. 


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



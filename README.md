# QAMD:Quality-Aware Blind Image Motion Deblurring
This is the source code for [QAMD:Quality-Aware Blind Image Motion Deblurring](https://www.sciencedirect.com/science/article/abs/pii/S0031320324003194).![KG-IQA Framework](https://github.com/esnthere/QAMD/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 2.1.2
CUDA: 12.1
Python: 3.11

## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k with 5%, 10%, 25%, 80% samples are released. The files in the folder of '**clip'** are obtained from open accessed source code of [CoOP](https://github.com/KaiyangZhou/CoOp) . 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/111iPWcQ7baaC5b771ZQ3Aw?pwd=j7pq). Please download these files and put them in the same folder of code and then run '**test_koniq_rt'*n*'.py**' to make intra/cross dataset test for models trained on *n%* samples.
   
   
## For train:  
The training code can be available at the 'training' folder.


## If you like this work, please cite:

{

  author={Tianshu Song, Leida Li, Jinjian Wu, Weisheng Dong, Deqiang Cheng,},
  
  journal={Pattern Recognition}, 
  
  title={Quality-aware blind image motion deblurring}, 
  
  year={2024},  
  
  doi={10.1016/j.patcog.2024.110568}
  
}

  
## License
This repository is released under the Apache 2.0 license. 


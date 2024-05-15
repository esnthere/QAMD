# QAMD:Quality-Aware Blind Image Motion Deblurring
This is the source code for [QAMD:Quality-Aware Blind Image Motion Deblurring](https://www.sciencedirect.com/science/article/abs/pii/S0031320324003194).![KG-IQA Framework](https://github.com/esnthere/QAMD/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 2.1.2
CUDA: 12.1
Python: 3.11

## For test:
### 1. Motion deblurring 
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/111iPWcQ7baaC5b771ZQ3Aw?pwd=j7pq). Please download the file and put them in the same folder of code, and then create a 'results' folder with subfolders of each dataset (such as 'Hide'). Finally, run '**deblurring.py**' for motion deblurring for each dataset, and the deblurred images will be saved in the subfolder.
   
### 2. Calculate values of DISTS and LPIPS  
   The files in the folder of '**lpips'** are obtained from open accessed source code of [LPIPS](https://github.com/KaiyangZhou/CoOp) and the '**DISTS_pt.py**' is modified from  open accessed source code of [DISTS](https://github.com/KaiyangZhou/CoOp) . To calculate values of DISTS and LPIPS, please run  '**test_crossdatasets_dists_lpips.py**' and a 'mat' file containing all values of DISTS and LPIPS will be saved.
   
  
   
   
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


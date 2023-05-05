# Hyperspectral anomaly change detection based on autoencoder
Pytorch implementation of JSTARS paper "Hyperspectral anomaly change detection based on autoencoder".
![image](https://github.com/meiqihu/ACDA/blob/main/Figure_ACDA.png)
# Paper
[Hyperspectral anomaly change detection based on autoencoder](https://ieeexplore.ieee.org/document/9380336)

Please cite our paper if you find it useful for your research.

>@ARTICLE{9380336,
  author={Hu, Meiqi and Wu, Chen and Zhang, Liangpei and Du, Bo},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Hyperspectral Anomaly Change Detection Based on Autoencoder}, 
  year={2021},
  volume={14},
  number={},
  pages={3750-3762},
  doi={10.1109/JSTARS.2021.3066508}}

# Installation
Install Pytorch 1.10.2 with Python 3.6
# Dataset
Download the [dataset of Viareggio 2013]
链接：https://pan.baidu.com/s/1x_M0nRqV-jmugIB6MltmXQ 
提取码：ogum

[Dataset]: "Viareggio 2013" with de-striping, noise-whitening and spectrally binning

>img_data.mat:  

>>img_1(D1F12H1);   img_2(D1F12H2);    img_3(D2F22H2)

>pretrain_samples:     

>>un_idx_train1,un_idx_valid1,un_idx_train2,un_idx_valid2;  [acquired from the pre-detection result of USFA, Wu C, Zhang L, Du B. Hyperspectral anomaly change detection with slow feature analysis[J]. Neurocomputing, 2015, 151: 175-187.]

>groundtruth_samples:  

>>un_idx_train1,un_idx_valid1,un_idx_train2,un_idx_valid2;

>random_samples:     un_idx_train1,un_idx_valid1,un_idx_train2,un_idx_valid2;
# Usage
maincode.py

# More
🌷[Homepage](https://meiqihu.github.io/)🌷  </br>
🔴[Google web](https://scholar.google.com.hk/citations?hl=zh-CN&user=jxyAHdkAAAAJ) 🔴 </br>
🌏[ResearchGate](https://www.researchgate.net/profile/Humeiqi-humeiqi) 🌍






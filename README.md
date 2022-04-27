# ACDA
Pytorch code of "Hyperspectral Anomaly Change Detection Based on Auto-encoder"
try "maincode.py"

This is a code of the paper "Hyperspectral anomaly change detection based on autoencoder" implemented on PyTorch.
Pytorch is needed for running this code.

-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
[Dataset]: "Viareggio 2013" with de-striping, noise-whitening and spectrally binning
1.   img_data.mat:    
img_1(D1F12H1);   img_2(D1F12H2);    img_3(D2F22H2)

链接：https://pan.baidu.com/s/1sRmdjsT-xl6DQJeoPIBNYA 
提取码：qdqf


2.   pretrain_samples:     
un_idx_train1,un_idx_valid1,un_idx_train2,un_idx_valid2;  [acquired from the pre-detection result of USFA, Wu C, Zhang L, Du B. Hyperspectral anomaly change detection with slow feature analysis[J]. Neurocomputing, 2015, 151: 175-187.]
3.  groundtruth_samples:     
un_idx_train1,un_idx_valid1,un_idx_train2,un_idx_valid2;
4.   random_samples:     un_idx_train1,un_idx_valid1,un_idx_train2,un_idx_valid2;

-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
[Usage]:  maincode.py

-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
If you use this code for your research, please cite our papers:
Hu M, Wu C, Zhang L, et al. Hyperspectral anomaly change detection based on autoencoder[J]. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021, 14: 3750-3762.

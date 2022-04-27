# coding=utf-8
import numpy as np
import torch
import time
import scipy.io as sio
from PIL import Image
import os
import models
import common_func
import Train_func


def train(x, y, idx_tra, idx_vld,h1, h2, lrn_rt,w_decay,ground_truth):
    model_1 = models.AutoEncoder(in_dim=127, hid_dim1=h1, hid_dim2=h2)
    model_2 = models.AutoEncoder(in_dim=127, hid_dim1=h1, hid_dim2=h2)
    model_1.apply(common_func.initNetParams)
    model_2.apply(common_func.initNetParams)  # function：initNetParams
    epoch, bth_sz, = 200, 256
    use_gpu = torch.cuda.is_available()  # False
    model_1, ls_m1, prdt_y, T_ls_1, V_ls_1 = Train_func.train_predict(
        x, y, model_1, idx_tra, idx_vld, epoch, bth_sz, lrn_rt, w_decay,use_gpu)
    model_2, ls_m2, prdt_x, T_ls_2, V_ls_2 = Train_func.train_predict(
        y, x, model_2, idx_tra, idx_vld, epoch, bth_sz, lrn_rt, w_decay,use_gpu)
    loss_result = np.minimum(ls_m1, ls_m2)
    X, Y, auc = common_func.plot_roc(loss_result.transpose(), ground_truth)
    print("auc is ", auc,'\n')
    return loss_result,ls_m1,ls_m2,prdt_y,prdt_x


if __name__ == '__main__':
    start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    use_gpu = torch.cuda.is_available()

    # Step1 : Read Data
    path_name = '/data/meiqi.hu/PycharmProjects/HyperspectralACD/AE/ACDA/'
    EX, img_2, train_smp, valid_smp = 'EX1', 'img_2', 'un_idx_train1', 'un_idx_valid1'
    ground_truth = Image.open(path_name + 'ref_EX1.bmp')
    if EX == 'EX2':
        img_2, train_smp, valid_smp = 'img_3', 'un_idx_train2', 'un_idx_valid2'
        ground_truth = Image.open(path_name + 'ref_EX2.bmp')
    # read image data
    # img_data : img_1,img_2,img_3(de-striping, noise-whitening and spectrally binning)
    data_filename = 'img_data.mat'
    data = sio.loadmat(path_name + data_filename)
    img_x0 = data['img_1']
    img_y0 = data[img_2]
    input_x = img_x0.transpose((2, 1, 0))
    input_y = img_y0.transpose((2, 1, 0))
    # read pre-train samples from pretraining result of USFA
    # for different training strategy(only replace the training samples)
    TrainSmp_filename = 'groundtruth_samples.mat' # groundtruth_samples random_samples pretrain_samples
    TrainSmp = sio.loadmat(path_name + TrainSmp_filename)
    un_idx_train = TrainSmp[train_smp].squeeze()
    un_idx_valid = TrainSmp[valid_smp].squeeze()
    img_channel, img_height, img_width = input_x.shape

    # Step2 : for experiemntal result
    Loss_result = np.zeros([img_height, img_width], dtype=float)
    h1, h2 = 60, 40  # 127, 127
    learn_rate, w_decay = 0.001, 0.001
    iter = 1
    Loss_result = np.zeros([img_height, img_width], dtype=float)
    for i in np.arange(1, 1 + iter):
        print('epoch i =', i)
        loss_result, ls_m1, ls_m2, prdt_y, prdt_x = train(input_x, input_y, un_idx_train, un_idx_valid, h1, h2,
                                                          learn_rate, w_decay,ground_truth)
        Loss_result = Loss_result + loss_result
    Loss_result = Loss_result / iter
    X, Y, auc = common_func.plot_roc(Loss_result.transpose(), ground_truth)

    print("auc is ", auc, '\n')
    print("-------------Ending---------------")
    print("     ")
    print(EX)
    end = time.time()
    print("共用时", (end - start), "秒")
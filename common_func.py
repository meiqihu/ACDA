# coding=utf-8
import numpy as np
from torch import nn
from torch.nn import init


def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.001)
            #init.xavier_uniform(m.weight)
            init.constant(m.bias, 0)
            #if m.bias:

        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.fill_(0)


# for evaluating the performance of the anomaly change detection result
def plot_roc(predict, ground_truth):
    """
    INPUTS:
     predict - anomalous change intensity map
     ground_truth - 0or1
    OUTPUTS:
     X, Y for ROC plotting
     auc
    """
    max_value = np.max(ground_truth)
    if max_value != 1:
        ground_truth = ground_truth / max_value

    # initial point（1.0, 1.0）
    x = 1.0
    y = 1.0
    hight_g, width_g = ground_truth.shape
    hight_p, width_p = predict.shape
    if hight_p != hight_g:
        predict = np.transpose(predict)

    ground_truth = ground_truth.reshape(-1)
    equals_one1 = np.where(ground_truth == 1)
    predict = predict.reshape(-1)
    # compuate the number of positive and negagtive pixels of the ground_truth
    pos_num = np.sum(ground_truth == 1)
    neg_num = np.sum(ground_truth == 0)
    # step in axis of  X and Y 
    x_step = 1.0 / neg_num
    y_step = 1.0 / pos_num
    # ranking the result map 
    index = np.argsort(list(predict))
    # predict = sorted(predict)
    ground_truth = ground_truth[index]
    equals_one2 = np.where(ground_truth == 1)
    """ 
    for i in ground_truth:
     when ground_truth[i] = 1, TP minus 1，one y_step in the y axis, go down
     when ground_truth[i] = 0, FP minus 1，one x_step in the x axis, go left
    """
    X = np.zeros(ground_truth.shape)
    Y = np.zeros(ground_truth.shape)
    for idx in range(0, hight_g * width_g):
        if ground_truth[idx] == 1:
            y = y - y_step
        else:
            x = x - x_step
        X[idx] = x
        Y[idx] = y

    auc = -np.trapz(Y, X) 
    if auc < 0.5:
        auc = -np.trapz(X, Y)
        t = X
        X = Y
        Y = t

    return X, Y, auc



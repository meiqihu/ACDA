# coding=utf-8
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset


def train_predict(x_3d, y_3d, model, un_idx_train, un_idx_valid,
                      epoch, batch_sz, learn_rate, w_decay,use_gpu):
    img_channel, img_height, img_width = x_3d.shape
    # x_3d  :  [img_channel,img_height,img_width]
    x_2d = np.transpose(x_3d.reshape(img_channel, img_height * img_width))  # [num,band]
    y_2d = np.transpose(y_3d.reshape(img_channel, img_height * img_width))
    train_dataset = TensorDataset(torch.tensor(x_2d[un_idx_train, :], dtype=torch.float32),
                                  torch.tensor(y_2d[un_idx_train, :], dtype=torch.float32))
    valid_label_x = torch.tensor(x_2d[un_idx_valid, :], dtype=torch.float32)
    valid_label_y = torch.tensor(y_2d[un_idx_valid, :], dtype=torch.float32)
    data_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    iter_num = un_idx_train.size // batch_sz
    loss_fc = nn.MSELoss()
    # While constructing the network, transfer the model to GPU (pytorch)
    if (use_gpu):
        model = model.cuda()
        loss_fc = loss_fc.cuda()

    optimizer = torch.optim.Adam \
        (model.parameters(), lr=learn_rate, betas=(0.9, 0.99), weight_decay=w_decay)

    # Training loss  &  Valid loss
    Tra_ls, Val_ls = [], []
    for _epoch in range(0, epoch):
        model.train()
        tra_ave_ls = 0
        for i, data in enumerate(data_loader):
            train_x, train_y = data
            # While traning, transfer the data to GPU
            if (use_gpu):
                train_x, train_y = train_x.cuda(), train_y.cuda()
            predict_y = model(Variable(train_x))
            loss = loss_fc(model(Variable(train_x)), train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tra_ave_ls += loss.item()
            tra_ave_ls /= iter_num
        Tra_ls.append(tra_ave_ls)
        model.eval()
        if (use_gpu):
            valid_label_x, valid_label_y = valid_label_x.cuda(), valid_label_y.cuda()
        val_ls = loss_fc(model(valid_label_x), valid_label_y).item()
        Val_ls.append(val_ls)
    # print('epoch [{}/{}],train:{:.4f},  valid:{:.4f}'.
    #     format(_epoch + 1, epoch, tra_ave_ls, val_ls))
    # # if _epoch % 5 == 0 :  print('epoch [{}/{}],train:{:.4f},  valid:{:.4f}'.
    #                           format(_epoch + 1, epoch, tra_ave_ls,val_ls))

    # Prediction
    model.eval()
    x_2d = torch.tensor(x_2d, dtype=torch.float32)
    if (use_gpu):
        x_2d = x_2d.cuda()
    prediction_y = model(x_2d)  # [num, band]
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    if (use_gpu):
        loss_fn = loss_fn.cuda()
    input_y = torch.autograd.Variable(torch.from_numpy(y_2d)).float()  # [num, band]
    if (use_gpu):
        input_y = input_y.cuda()
    loss = loss_fn(input_y, prediction_y)
    if (use_gpu):
        loss,prediction_y= loss.cpu(),prediction_y.cpu()
    loss_m1 = np.sum(loss.detach().numpy(), axis=1).reshape(img_height, img_width)  # axis=1,[num, 1]
    prediction_y = prediction_y.detach().numpy().transpose(). \
        reshape([img_channel, img_height, img_width, ])

    return model, loss_m1, prediction_y, Tra_ls, Val_ls

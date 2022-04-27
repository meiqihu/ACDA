#coding=utf-8

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('/data/meiqi.hu/PycharmProjects/HyperspectralACD/MyFunction/')
import common_func
from PIL import Image
import random



class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim1,hid_dim2):
        super(AutoEncoder, self).__init__()
        self.out_dim = in_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim2, bias=True),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim2, hid_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim1, in_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        feature = self.encoder(x)
        cons_x = self.decoder(feature)
        return cons_x

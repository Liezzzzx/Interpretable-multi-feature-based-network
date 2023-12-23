#coding=gbk 
'''
just talk about age above 19
'''

import numpy as np
import torch,gc
import scipy.ndimage as nd
import nibabel as nib
from nibabel import nifti1
import torch.utils.data as Data
import torch.nn as nn
import os
import pandas as pd
from torch.optim import lr_scheduler

class C3D(nn.Module):
    """
    The C3D network.
    """
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1a = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv1b = nn.Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.batch_norm1 = nn.BatchNorm3d(num_features=8)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2a = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv2b = nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.batch_norm2 = nn.BatchNorm3d(num_features=16)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv3b = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.batch_norm3 = nn.BatchNorm3d(num_features=32)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv4b = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.batch_norm4 = nn.BatchNorm3d(num_features=64)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv5a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv5b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.batch_norm5 = nn.BatchNorm3d(num_features=128)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(1536, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 1)
        # self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = torch.as_tensor(x,dtype=torch.float32)

        x = self.relu(self.conv1a(x))
        x = self.conv1b(x)
        x = self.relu(self.batch_norm1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2a(x))
        x = self.conv2b(x)
        x = self.relu(self.batch_norm2(x))
        x = self.pool2(x)


        x = self.relu(self.conv3a(x))
        x = self.conv3b(x)
        x = self.relu(self.batch_norm3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.conv4b(x)
        x = self.relu(self.batch_norm4(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.conv5b(x)
        x = self.relu(self.batch_norm5(x))
        x = self.pool5(x)

        x = x.view(x.size(0),-1)
        x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x




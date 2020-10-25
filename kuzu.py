# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.l = nn.Linear(28*28, 10)

    def forward(self, x):
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        o = self.l(x)
        o = F.log_softmax(o)
        return o # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(28*28, 500)
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        o1 = torch.tanh(self.l1(x))
        o = F.log_softmax(self.l2(o1))
        return o # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):  
        #x = x.view(x.shape[0], -1)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2) 
        #print(x.shape)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        o = F.log_softmax(x)
        return o # CHANGE CODE HERE

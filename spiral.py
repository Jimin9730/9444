# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(2, num_hid)
        self.l2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt(x**2 + y**2).reshape(-1, 1)
        a = torch.atan2(y, x).reshape(-1, 1)
        cat_input = torch.cat((r, a), 1)
        hid_l1 = self.l1(cat_input)
        self.h1 = torch.tanh(hid_l1)
        hid_l2 = self.l2(self.h1)        
        self.output = torch.sigmoid(hid_l2)
        return self.output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(2, num_hid)
        self.l2 = nn.Linear(num_hid, num_hid)
        self.l3 = nn.Linear(num_hid, 1)

    def forward(self, input):
        hid_l1 = self.l1(input)
        self.h1 = torch.tanh(hid_l1)
        hid_l2 = self.l2(self.h1)
        self.h2 = torch.tanh(hid_l2)
        hid_l3 = self.l3(self.h2)
        self.output = torch.sigmoid(hid_l3)
        return self.output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        if layer == 1:
            pred = (net.h1[:, node] >= 0).float()
        elif layer == 2:
            pred = (net.h2[:, node] >= 0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
  

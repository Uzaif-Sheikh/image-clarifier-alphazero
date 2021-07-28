#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.ToTensor()
    elif mode == 'test':
        return transforms.ToTensor()

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(3,32,3,padding=1)
        self.con2 = nn.Conv2d(32,54,3,padding=1)
        self.con3 = nn.Conv2d(54,64,3,padding=1)
        self.batch = nn.BatchNorm2d(32) 
        self.batch1 = nn.BatchNorm2d(64) 
        self.max = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(64*8*8,1024)
        self.linear2 = nn.Linear(1024,564)
        self.output = nn.Linear(564,14)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, t):
        t = self.max(F.relu(self.con1(t)))
        t = self.batch(t)
        t = self.max(F.relu(self.con2(t)))
        t = self.max(F.relu(self.con3(t)))
        t = self.batch1(t)
        t = t.view(t.shape[0],-1)
        t = F.relu(self.linear1(t))
        t = self.dropout(t)
        t = F.relu(self.linear2(t))
        #t = self.dropout(t)
        t = self.output(t)
        t = F.log_softmax(t,dim=1)
        return t

class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
    """
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        lossfunc = nn.CrossEntropyLoss()
        loss = lossfunc(output,target)
        return loss



net = Network()
lossFunc = loss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 256
epochs = 10
optimiser = optim.Adam(net.parameters(), lr=0.001)

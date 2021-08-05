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
alphazero comp9444 
Kartikaye Chandhok (z5285022)
Uzaif Sheikh (z5252826)
g024898
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
The Neural Network we designed consists of five convolutional layer with increasing output
feature channels from each one starting from 3 channels to 52 in the first convolutional channel,
52 to 96 channels in the second convolutional layer, 96 to 124 feature channels in the third convolutional layer,
124 to 175 output features in the fourth convolutional layer and lastly 175 input features to 250 output features
in the fifth convolutional layer. ReLu activation is used in each convolutional layer to sharpen the feature maps.
We also apply batch normalization after each convolutional layer along with dropouts of 0.3 on the second,
third and fourth convolutional layer to normalize and regularize the data to prevent overfitting and speed up the learning process.
Kernel size is 3 x 3 for each layer and a padding of 1 is added so kernel is able to fit over the image properly.
The images from each convolutional layer excluding the first one are also pooled down subsequently to a 4 x 4 image,
this is done to reduce the image dimension size as we increase the number of feature maps and also to abstract more features
from the convolutional layers.
The fully linear layer consists of two hidden layer with 250 x 4 x 4 nodes in the input layer, 764 nodes in the first 
hidden layer and 254 nodes in the secong hidden layer with Relu activation function in each node of the hidden layers.
We use a dropout of 0.5 on the two hidden layers to help prevent overfitting.
The output layer consists of 14 nodes for classifying the 14 characters and log_softmax function for multiclass
classification.
As we are using log softmax as our activation function in our output layer which classifies the images using predicted output probabilities,
we chose Cross Entropy Loss function as our loss function which works well with softmax to minimize the loss between actual
and predicted probabilities when working with muliclass classification.
Optimizer used is Adam.
The training and validation split was changed to 0.9 to provide more data for the trainig phase to prevent overfitting and
the model was trained for 50 epochs. The learning rate and batch size remain unchanged. 
We started our model with two convolutional layer and a fully connected linear layer with one hidden layer, the simplest 
idea to see how our model performed. It achieved around 30% accuracy on the trainig phase and 25% in the validation set.
We started making our model architecture more complicated and observed that adding more convolutional layers increased the
training accuracy but validation set accuracy remained the same. We realized the model was overfitting, so we decided to 
introduce some regularization methods like adding dropout layers and batch normalization, we set the batch normalization
across all the convolutional layers and set the dropuout values to deafult, which seemed too much so we reduced it around 0.3, we 
applied max pooling to four convolutional layer rather than applying it to every layer as that will change the size of the 
image to 2 x 2 which leads in lower accuracy as the features of the image was not clear due to the lower pixel size. Lastly in 
order to reduce overfitting we increased the training set from 0.8 to 0.90-0.95 and when we tried to change the optimizer from Adam 
to SGD which lowered our accuracy, we decided to stick with Adam. The architecture described above is our final model and 
our current accuracy is 98 on the training set and 90-93 on the validation set.

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
        self.con1 = nn.Conv2d(3,52,3,padding=1)
        self.con2 = nn.Conv2d(52,96,3,padding=1)
        self.con3 = nn.Conv2d(96,124,3,padding=1)
        self.con4 = nn.Conv2d(124,175,3,padding =1)
        self.con5 = nn.Conv2d(175, 250, 3, padding = 1)
        self.batch = nn.BatchNorm2d(52) 
        self.batch2 = nn.BatchNorm2d(96)
        self.batch1 = nn.BatchNorm2d(175) 
        self.batch3 = nn.BatchNorm2d(124)
        self.batch5 = nn.BatchNorm2d(250)
        self.max = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(250*4*4,764)
        #self.linear2 = nn.Linear(3064,1532)
        self.linear3 = nn.Linear(764,254)
        self.output = nn.Linear(254,14)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_conv = nn.Dropout(p=0.3)
        
    def forward(self, t):
        t = F.relu(self.con1(t))
        t = self.batch(t)

        t = self.max(F.relu(self.con2(t)))
        t = self.batch2(t)
        t = self.dropout_conv(t)

        t = self.max(F.relu(self.con3(t)))
        t = self.batch3(t)

        t = self.max(F.relu(self.con4(t)))
        t = self.batch1(t)
        t = self.dropout_conv(t)

        t = self.max(F.relu(self.con5(t)))
        t = self.batch5(t)
        t = self.dropout_conv(t)

        t = t.view(t.shape[0],-1)
        t = self.dropout1(t)
        t = F.relu(self.linear1(t))
        t = self.dropout(t)
        #t = F.relu(self.linear2(t))
        #t = self.dropout(t)
        t = F.relu(self.linear3(t))
        t = self.dropout(t)
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
train_val_split = 0.95
batch_size = 256
epochs = 40
optimiser = optim.Adam(net.parameters(), lr=0.001)
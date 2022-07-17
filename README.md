# Alphazero

Alphazero a image clarifer to classify Simpson characters, here is a breif explanation on the chosen 
Neural Network:

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

```python
        # creating convolutional layer from 3 to 52 with kernel size 3 and padding 1. 
        self.con1 = nn.Conv2d(3,52,3,padding=1)

        # Conv layer of 52 to 96 with kernel size 3 and padding 1.
        self.con2 = nn.Conv2d(52,96,3,padding=1)

        # Conv layer of 96 to 124 with kernel size 3 and padding 1.
        self.con3 = nn.Conv2d(96,124,3,padding=1)

        # Conv layer of 124 to 175 with kernel size 3 and padding 1.
        self.con4 = nn.Conv2d(124,175,3,padding =1)

        # Conv layer of 175 to 250 wither kernel size 3 and padding 1.
        self.con5 = nn.Conv2d(175, 250, 3, padding = 1)
```

The fully linear layer consists of two hidden layer with 250 x 4 x 4 nodes in the input layer, 764 nodes in the first 
hidden layer and 254 nodes in the secong hidden layer with Relu activation function in each node of the hidden layers.
We use a dropout of 0.5 on the two hidden layers to help prevent overfitting.
The output layer consists of 14 nodes for classifying the 14 characters and log_softmax function for multiclass
classification.
As we are using log softmax as our activation function in our output layer which classifies the images using predicted output probabilities,
we chose Cross Entropy Loss function as our loss function which works well with softmax to minimize the loss between actual
and predicted probabilities when working with muliclass classification.
Optimizer used is Adam.

```python
        # doing batch normlization for respective out features.
        self.batch = nn.BatchNorm2d(52) 
        self.batch2 = nn.BatchNorm2d(96)
        self.batch1 = nn.BatchNorm2d(175) 
        self.batch3 = nn.BatchNorm2d(124)
        self.batch5 = nn.BatchNorm2d(250)

        # creating max pooling layer of size 2 x 2.
        self.max = nn.MaxPool2d(2,2)

        # Creating linear layer from 250*4*4 to 764
        self.linear1 = nn.Linear(250*4*4,764)

        # Creating linear layer from 764 to 254
        self.linear3 = nn.Linear(764,254)

        # Creating output layer from 254 to 14
        self.output = nn.Linear(254,14)

        # Creating dropout layer of 0.2, 0.5 and 0.3
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_conv = nn.Dropout(p=0.3)
```

The training and validation split was changed to 0.9-0.95 to provide more data for the trainig phase to prevent overfitting and
the model was trained for 40 epochs. The learning rate and batch size remain unchanged. 
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
our current accuracy is 98% on the training set and 90-93% on the validation set.
 

## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)  
        #32x110x110
        self.conv2 = nn.Conv2d(32, 64, 3)
        #64x54x54
        self.conv3 = nn.Conv2d(64, 128,  3)
        #128x26x26
        self.conv4 = nn.Conv2d(128, 256, 2)
        #256x12x12

        #pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        #linear layers
        self.fc1 = nn.Linear( 256*12*12, 500) 
        self.fc2 = nn.Linear(500, 136)

        # Dropout
        self.dropout = nn.Dropout(p = 0.3)
        self.dropout2 = nn.Dropout(p=0.2)
        




    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))

        x = self.dropout(self.pool(F.relu(self.conv2(x))))

        x = self.dropout(self.pool(F.relu(self.conv3(x))))


        x = self.dropout(self.pool(F.relu(self.conv4(x))))

        # Flattening the layer
        x = x.view(x.size(0), -1)

        x = self.dropout2(F.relu(self.fc1(x)))

        x = self.fc2(x)
        return x

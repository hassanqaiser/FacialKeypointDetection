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
        

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 110
        self.conv2 = nn.Conv2d(32, 64, 3)
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 54
        self.conv3 = nn.Conv2d(64, 128, 3)
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 26
        self.conv4 = nn.Conv2d(128, 256, 3)
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 12
        
        # Maxpool
        self.pool = nn.MaxPool2d(2, 2)    
        
        # Fully Connected
        # 256*12*12 = 36864
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)

        # BatchNorm 6,20,32,64
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        
        # Dropouts
        self.drop1 = nn.Dropout2d(p = 0.1)
        self.drop2 = nn.Dropout2d(p = 0.2)
        self.drop3 = nn.Dropout2d(p = 0.3)

        
        

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        #print("First size: ", x.shape)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop1(x)
        #print("Second size: ", x.shape)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop2(x)
        #print("Third size: ", x.shape)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.drop2(x)
        #print("Fourth size: ", x.shape)
        
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)
        
        x = F.relu(self.fc1(x))
        #print("First FC size: ", x.shape)

        x = self.drop3(x)

        x = F.relu(self.fc2(x))
        #print("Second FC size: ", x.shape)

        x = self.drop3(x)

        x = self.fc3(x)
        #print("Third FC size: ", x.shape)
        
        #x = F.log_softmax(x, dim=1)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

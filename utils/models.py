# Basic implementation of both SDKN and NN without using the pytorch-lightning framework.

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import kernels
from utils.utilities import ActivFunc


# Implementation of an exemplary SDKN model
class SDKN(torch.nn.Module):  # inherit from the StandardDK class to have the same init
    def __init__(self, centers, dim_input, dim_output):
        super(SDKN, self).__init__()

        # General stuff
        self.centers = centers
        self.M = self.centers.shape[0]

        # Define linear maps
        self.width = 10
        self.fc1 = nn.Linear(dim_input, self.width, bias=False)
        self.fc2 = nn.Linear(self.width, self.width, bias=False)
        self.fc3 = nn.Linear(self.width, dim_output, bias=False)

        # Define activation maps
        self.activ1 = ActivFunc(self.width, self.M, kernel=kernels.Gaussian(ep=1))
        self.activ2 = ActivFunc(self.width, self.M, kernel=kernels.Wendland_order_0(ep=1))

    def forward(self, x):
        centers = self.centers

        # First fully connect + activation function
        x = self.fc1(x)
        centers = self.fc1(centers)
        x, centers = self.activ1(x, centers)

        # Second fully connect + activation function
        x = self.fc2(x)
        centers = self.fc2(centers)
        x, centers = self.activ2(x, centers)

        # Third fully connect
        x = self.fc3(x)

        return x

# Implementation of an exemplary NN model
class NN(nn.Module):

    def __init__(self, dim_input, dim_output):
        super(NN, self).__init__()

        self.width = 15

        self.fc1 = nn.Linear(dim_input, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, dim_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
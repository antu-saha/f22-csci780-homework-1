import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Define 1st convolution layer
        self.conv1 = nn.Conv2d(3, 16, 2)  # input_channel, output_channel, kernel_size
        # Initialize weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.conv1.weight)
        # Define batch normalization
        self.conv1_bn = nn.BatchNorm2d(16)

        # Define 2nd convolution layer
        self.conv2 = nn.Conv2d(16, 32, 2)
        # Initialize weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.conv2.weight)
        # Define batch normalization
        self.conv2_bn = nn.BatchNorm2d(32)

        # Define 3rd convolution layer
        self.conv3 = nn.Conv2d(32, 64, 2)
        # Initialize weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.conv3.weight)
        # Define batch normalization
        self.conv3_bn = nn.BatchNorm2d(64)

        # Define Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size, stride

        # Define 1st fully connected layer
        self.fc1 = nn.Linear(64 * 3 * 3, 120)
        # Initialize weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.fc1.weight)
        # Define batch normalization
        self.fc1_bn = nn.BatchNorm1d(120)

        # Define 2nd fully connected layer
        self.fc2 = nn.Linear(120, 10)
        # Initialize weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.fc1.weight)

        # Define proportion of neurons to dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        # print(x.shape)
        # Apply Dropout
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        # print(x.shape)
        # Apply Dropout
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.pool(F.relu(self.conv3_bn(x)))
        # print(x.shape)
        # Flatten input
        x = x.view(-1, 64 * 3 * 3)
        # print(x.shape)

        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        # print(x.shape)
        # Apply Dropout
        x = self.dropout(x)
        x = self.fc2(x)
        # print(x.shape)
        return x

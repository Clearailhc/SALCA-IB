import torch
import torch.nn as nn

class DynamicCNN(nn.Module):
    def __init__(self, input_size, conv_channels, kernel_size, fc_units, output_size):
        super(DynamicCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_channels * ((input_size - kernel_size + 1) // 2), fc_units)
        self.fc2 = nn.Linear(fc_units, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


import torch
from torch import nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.mp = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.mp(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.mp(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(in_size, -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

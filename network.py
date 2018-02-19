import torch
import torch.nn as nn

    
class FCC(nn.Module):
    def __init__(self):
        super(FCC, self).__init__()
        self.BN0 = nn.BatchNorm1d(28*28)
        self.hidden0 = nn.Linear(28*28, 1024)
        self.BN1 = nn.BatchNorm1d(1024)
        self.hidden1 = nn.Linear(1024, 1024)
        self.BN2 = nn.BatchNorm1d(1024)
        self.hidden2 = nn.Linear(1024, 10)

        
    def forward(self, x):
        R, C, H, W = x.size()
        x = x.resize(R, C*H*W)
        
        x = self.BN0(x)
        x = self.hidden0(x)

        x = self.BN1(x)
        x = self.hidden1(x)

        x = self.BN2(x)
        x = self.hidden2(x)

        return x

    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        kernel_size = 5
        channels = 32
        self.in_channels = 1
        
        conv_layers = []
        for _ in range(6):
            conv_layers += [nn.BatchNorm2d(self.in_channels)]
            conv_layers += [nn.Conv2d(self.in_channels, channels, kernel_size)]
            conv_layers += [nn.ReLU(inplace = True)]
            self.in_channels = channels
        self.conv_layers = nn.Sequential(*conv_layers)

        self.dense0 = nn.Linear(4*4*32, 4*4*32)
        self.dense1 = nn.Linear(4*4*32, 10)

        
    def forward(self, x):
        R, _, _, _ = x.size()
        x = self.conv_layers(x)

        x = x.resize(R, 4*4*32)
        x = self.dense0(x)
        x = self.dense1(x)

        return x

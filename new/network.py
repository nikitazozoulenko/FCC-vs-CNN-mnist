import torch
import torch.nn as nn

class FCC(nn.Module):
    def __init__(self):
        super(FCC, self).__init__()
        
        layers = [nn.BatchNorm1d(28*28),
                  nn.Linear(28*28, 128),
                  nn.ReLU(inplace = True)]
        
        for _ in range(4):
            layers += [nn.BatchNorm1d(128),
                       nn.Linear(128, 128),
                       nn.ReLU(inplace = True)]

        layers += [nn.BatchNorm1d(128),
                   nn.Linear(128, 10)]
        
        self.fcc = nn.Sequential(*layers)

        
    def forward(self, x):
        R, _, _= x.size()
        x = x.view(R, -1)
        
        x = self.fcc(x)
        return x

    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        layers = []
        maxpool = nn.MaxPool2d(2, stride=2)

        layers += [self._make_BN_conv_relu(1, 1, 64)]
        layers += [self._make_BN_conv_relu(1, 64, 64)]
        layers += [maxpool]
        layers += [self._make_BN_conv_relu(3, 64, 64)]
        layers += [maxpool]
        layers += [self._make_BN_conv_relu(1, 64, 10)]

        self.conv_layers = nn.Sequential(*layers)
    

    def _make_BN_conv_relu(self, n_layers, in_channels, out_channels):
        layers = []
        for _ in range(n_layers):
            layers += [nn.BatchNorm2d(in_channels)]
            layers += [nn.Conv2d(in_channels, out_channels, 3, 1)]
            layers += [nn.ReLU()]
        return nn.Sequential(*layers)

        
    def forward(self, x):
        R, _, _= x.size()
        x = x.view(R, 1, 28, 28)
        x = self.conv_layers(x).view(R, -1)
        return x

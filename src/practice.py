import torch

import models

class Dtanh(torch.nn.Module):

    def forward(self, X):
        ans = torch.tanh(X)
        return ans * (1-ans**2)

class Model(models.Savable):
    def __init__(self, channels, classes, imagesize, **kwargs):
        super(Model, self).__init__()
        self.conv = torch.nn.Sequential(
        
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            Dtanh(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.BatchNorm2d(32),
            
            # 32 -> 16
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.BatchNorm2d(32),
            
            # 16 -> 8
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.BatchNorm2d(32),
            
            # 8 -> 4
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            Dtanh(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.AvgPool2d(4)
        )
        
        self.net = torch.nn.Linear(32, classes)
    
    def forward(self, X):
        return self.net(self.conv(X).squeeze())

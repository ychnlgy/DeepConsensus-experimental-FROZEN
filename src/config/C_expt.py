import torch

from .Base import Base

import models

class C_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
    
        initial_channels = 96 * 3
    
        return torch.nn.Sequential( # Parameter count: 162K
            
            # 28 -> 28
            models.ResNet(
                kernelseq = [3],
                headsize = channels,
                bodysize = initial_channels,
                tailsize = initial_channels,
                layers = 8
            ),
            
            # 28 -> 14
            torch.nn.Conv2d(initial_channels, 64, 3, padding=1, groups=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.Conv2d(64, 64, 3, padding=1, groups=4),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            torch.nn.Conv2d(64, 32, 3, padding=1, stride=2, groups=4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 4 -> 1
            torch.nn.AvgPool2d(4),
            
            models.Reshape(32),
            models.DenseNet(
                headsize = 32,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.1,
                bias = True
            )
        )

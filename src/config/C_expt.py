import torch

from .Base import Base

import models

class C_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
    
        core_channels = 64
        initial_channels = core_channels * channels
    
        return torch.nn.Sequential( # Parameter count: 49378
            
            # 28 -> 14
            torch.nn.Conv2d(channels, initial_channels, 3, padding=1, groups=channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(initial_channels),
            
            # 14 -> 7
            torch.nn.Conv2d(initial_channels, core_channels, 3, padding=1, groups=core_channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(core_channels),
            
            # 7 -> 4
            torch.nn.Conv2d(core_channels, 32, 3, padding=1, groups=4),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            
            # 4 -> 1
            torch.nn.Conv2d(32, 16, 3, padding=1, groups=4),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(4),
            torch.nn.BatchNorm2d(16),
            
            models.Reshape(16),
            models.DenseNet(
                headsize = 16,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.1,
                bias = True
            )
        )

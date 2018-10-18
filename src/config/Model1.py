import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.ResNet(
                kernelseq = [3, 3],
                headsize = channels,
                bodysize = 128,
                tailsize = 128,
                layers = 8
            ),
            
            # 28 -> 14
            torch.nn.Conv2d(128, 64, 3, padding=1, groups=4),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.Conv2d(64, 64, 3, padding=1, groups=4),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            torch.nn.Conv2d(64, 64, 3, padding=1, groups=4),
            torch.nn.AvgPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 1
            torch.nn.Conv2d(64, 64, 3, padding=1, groups=4),
            torch.nn.AvgPool2d(4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            models.Reshape(64),
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
            
        )

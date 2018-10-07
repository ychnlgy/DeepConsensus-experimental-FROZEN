import torch

from .Base import Base

import models

class C_150K(Base):

    @staticmethod
    def get_paramid():
        return "150K"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 150240
            
            # 28 -> 28
            torch.nn.Conv2d(channels, 64, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 28 -> 14
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=1),
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            # 14 -> 7
            torch.nn.Conv2d(128, 54, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(54),
            
            # 7 -> 4
            torch.nn.Conv2d(54, 20, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(20),
            
            # 4 -> 10
            torch.nn.Conv2d(20, classes, 4, padding=0),
            models.Reshape(len, classes)
        )

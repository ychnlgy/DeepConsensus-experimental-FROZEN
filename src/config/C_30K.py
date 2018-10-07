import torch

from .Base import Base

import models

class C_30K(Base):

    @staticmethod
    def get_paramid():
        return "30K"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 30134
            
            # 28 -> 14
            torch.nn.Conv2d(channels, 128, 3, padding=1, stride=1),
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            # 14 -> 7
            torch.nn.Conv2d(128, 20, 3, padding=1, stride=1),
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(20),
            
            # 7 -> 4
            torch.nn.Conv2d(20, 16, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 4 -> 10
            torch.nn.Conv2d(16, classes, 4, padding=0),
            models.Reshape(len, classes)
        )

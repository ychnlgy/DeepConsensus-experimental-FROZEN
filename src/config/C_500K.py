import torch

from .Base import Base

import models

class C_500K(Base):

    @staticmethod
    def get_paramid():
        return "500K"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 504852
            
            # 28 -> 28
            torch.nn.Conv2d(channels, 128, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            # 28 -> 14
            torch.nn.Conv2d(128, 256, 3, padding=1, stride=1),
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 14 -> 7
            torch.nn.Conv2d(256, 78, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(78),
            
            # 7 -> 4
            torch.nn.Conv2d(78, 32, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 4 -> 10
            torch.nn.Conv2d(32, classes, 4, padding=0),
            models.Reshape(len, classes)
        )

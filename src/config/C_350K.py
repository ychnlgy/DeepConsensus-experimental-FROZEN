import torch

from .Base import Base

import models

class C_350K(Base):

    @staticmethod
    def get_paramid():
        return "350K"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 344954
            
            # 28 -> 28
            torch.nn.Conv2d(channels, 512, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(512),
            
            # 28 -> 14
            torch.nn.Conv2d(512, 64, 3, padding=1, stride=1),
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.Conv2d(64, 36, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(36),
            
            # 7 -> 4
            torch.nn.Conv2d(36, 28, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(28),
            
            # 4 -> 10
            torch.nn.Conv2d(28, classes, 4, padding=0),
            models.Reshape(len, classes)
        )

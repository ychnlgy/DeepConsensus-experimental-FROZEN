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
            torch.nn.Conv2d(channels, 256, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 28 -> 14
            torch.nn.Conv2d(256, 256, 3, padding=1, stride=1, groups=256),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 14 -> 7
            torch.nn.Conv2d(256, 256, 3, padding=1, stride=2, groups=256),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 7 -> 4
            torch.nn.Conv2d(256, 256, 3, padding=1, stride=2, groups=256),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 4 -> 1
            models.Reshape(len, 256, 16, contiguous=True),
            models.Permute(0, 2, 1), # N, W*H, C
            models.Mean(dim=1), # N, C
            
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, classes),
            torch.nn.LeakyReLU(),
        )

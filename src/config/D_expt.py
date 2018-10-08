import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 50K
            
            # 28 -> 28
            models.DistillationLayer(
                cnn = torch.nn.Conv2d(channels, 32, 3, padding=1),
                lin = torch.nn.Linear(32, 32)
            ),
            
            # 28 -> 14
            models.DistillationLayer(
                cnn = torch.nn.Conv2d(32, 32, 3, padding=1, stride=2, groups=32),
                lin = torch.nn.Linear(32, 32)
            ),
            
            # 14 -> 7
            models.DistillationLayer(
                cnn = torch.nn.Conv2d(32, 32, 3, padding=1, stride=2, groups=32),
                lin = torch.nn.Linear(32, 32)
            ),
            
            # 7 -> 4
            models.DistillationLayer(
                cnn = torch.nn.Conv2d(32, 32, 3, padding=1, stride=2, groups=32),
                lin = torch.nn.Linear(32, 32)
            ),
            
            # 4 -> 1
            models.DistillationLayer(
                cnn = torch.nn.Conv2d(32, 32, 4, padding=0, stride=1, groups=32),
                lin = torch.nn.Linear(32, classes)
            ),
            
            models.Reshape(len, classes)
        )

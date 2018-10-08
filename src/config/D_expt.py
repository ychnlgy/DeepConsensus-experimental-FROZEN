import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 6474
            
            # 28 -> 28
            models.DistillationLayer(
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(channels, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                ),
                lin = torch.nn.Linear(128, 64)
            ),
            
            # 28 -> 14
            models.DistillationLayer(
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1, stride=2, groups=32),
                    torch.nn.BatchNorm2d(64),
                ),
                lin = torch.nn.Linear(64, 32)
            ),
            
            # 14 -> 7
            models.DistillationLayer(
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1, stride=2, groups=32),
                    torch.nn.BatchNorm2d(32),
                ),
                lin = torch.nn.Linear(32, 32)
            ),
            
            # 7 -> 4
            models.DistillationLayer(
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1, stride=2, groups=32),
                    torch.nn.BatchNorm2d(32),
                ),
                lin = torch.nn.Linear(32, 32)
            ),
            
            # 4 -> 1
            models.DistillationLayer(
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 4, padding=0, stride=1, groups=32),
                    torch.nn.BatchNorm2d(32),
                ),
                lin = torch.nn.Linear(32, classes)
            ),
            
            models.Reshape(len, classes)
        )

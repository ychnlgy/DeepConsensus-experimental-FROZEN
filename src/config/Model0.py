import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
            models.DistillNet(
            
                # 28 -> 14
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(channels, 48, 3, padding=1, groups=channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(48)
                    ),
                    pruner = models.Pruner(
                        delta = 1,
                        classes = classes
                    )
                ),
                
                # 14 -> 7
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(48, 32, 3, padding=1, groups=16),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pruner = models.Pruner(
                        delta = 1,
                        classes = classes
                    )
                ),
                
                # 7 -> 4
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 16, 3, padding=1, groups=16),
                        torch.nn.LeakyReLU(),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.BatchNorm2d(16)
                    ),
                    pruner = models.Pruner(
                        delta = 1,
                        classes = classes
                    )
                ),
            ),
            
            models.DenseNet(
                headsize = 48 + 32 + 16,
                bodysize = None,
                tailsize = classes,
                layers = 1,
                dropout = 0.0, # because the vector is distilled
                bias = True
            )
        )

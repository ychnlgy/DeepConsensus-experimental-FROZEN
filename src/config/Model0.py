import torch

import models

from .Base import Base

PRUNE = 3

class Model(Base):

    def create_net(self, channels, classes, delta):
        return torch.nn.Sequential(
            models.DistillNet(
            
                # 28 -> 14
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(channels, 64, 3, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        patience = PRUNE
                    )
                ),
                
                # 14 -> 7
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 128, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.Conv2d(128, 128, 3, padding=1, groups=128),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(128)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        patience = PRUNE
                    )
                ),
                
                # 7 -> 4
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(128, 64, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        patience = PRUNE
                    )
                ),
                
                # 4 -> 2
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                        torch.nn.LeakyReLU(),
                        torch.nn.AvgPool2d(2),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        patience = PRUNE
                    )
                ),
            ),
            
            models.DenseNet(
                headsize = 64 + 128 + 64 + 32,
                bodysize = 512,
                tailsize = classes,
                layers = 2,
                dropout = 0.1,
                bias = True
            )
        )

import torch

import models

from .Base import Base

PRUNE = 5

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
                        torch.nn.BatchNorm2d(64)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        prune_rest = PRUNE
                    )
                ),
                
                # 14 -> 7
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 128, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(128)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        prune_rest = PRUNE
                    )
                ),
                
                # 7 -> 4
                models.DistillBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(128, 64, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pruner = models.Pruner(
                        delta = delta,
                        classes = classes,
                        prune_rest = PRUNE
                    )
                ),
            ),
            
            models.DenseNet(
                headsize = 64 + 128 + 64,
                bodysize = 256,
                tailsize = classes,
                layers = 2,
                dropout = 0.0, # because the vector is distilled
                bias = True
            )
        )

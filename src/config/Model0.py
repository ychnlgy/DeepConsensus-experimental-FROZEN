import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
            models.DistillNet(
            
                # 28 -> 14
                models.DistillationBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(channels, 96, 3, padding=1, groups=channels),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(96)
                    ),
                    pool = models.SumPool(
                        paramsize = 128,
                        net = models.DenseNet(
                            headsize = 128,
                            bodysize = 256,
                            tailsize = 96,
                            layers = 2,
                            dropout = 0.0,
                            bias = True
                        ),
                        threshold = 0.02
                    )
                ),
                
                # 14 -> 7
                models.DistillationBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(96, 64, 3, padding=1, groups=32),
                        torch.nn.LeakyReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.BatchNorm2d(64)
                    ),
                    pool = models.SumPool(
                        paramsize = 64,
                        net = models.DenseNet(
                            headsize = 64,
                            bodysize = 128,
                            tailsize = 64,
                            layers = 2,
                            dropout = 0.0,
                            bias = True
                        ),
                        threshold = 0.02
                    )
                ),
                
                # 7 -> 4
                models.DistillationBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                        torch.nn.LeakyReLU(),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.BatchNorm2d(32)
                    ),
                    pool = models.SumPool(
                        paramsize = 32,
                        net = models.DenseNet(
                            headsize = 32,
                            bodysize = 64,
                            tailsize = 32,
                            layers = 2,
                            dropout = 0.0,
                            bias = True
                        ),
                        threshold = 0.02
                    )
                ),
            ),
            
            # distilled vector of 96 + 64 + 32
            models.DenseNet(
                headsize = 96 + 64 + 32,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.0, # because the vector is distilled
                bias = True
            )
        )

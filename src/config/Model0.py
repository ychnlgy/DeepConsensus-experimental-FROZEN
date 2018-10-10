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
                    pool = models.SumPool(
                        paramsize = 128,
                        net = models.DenseNet(
                            headsize = 128,
                            bodysize = 128,
                            tailsize = 48,
                            layers = 1,
                            dropout = 0.0,
                            bias = True
                        ),
                        threshold = 0.02
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
                    pool = models.SumPool(
                        paramsize = 64,
                        net = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            dropout = 0.0,
                            bias = True
                        ),
                        threshold = 0.02
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
                    pool = models.SumPool(
                        paramsize = 32,
                        net = models.DenseNet(
                            headsize = 32,
                            bodysize = 16,
                            tailsize = 16,
                            layers = 1,
                            dropout = 0.0,
                            bias = True
                        ),
                        threshold = 0.02
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

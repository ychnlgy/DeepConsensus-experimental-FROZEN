import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return models.DistillationNet(
            
            models.DistillationBlock(
            
                # 28 -> 14
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(channels, 96, 3, padding=1, groups=channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.BatchNorm2d(96)
                ),
                
                atn = models.AttentionPool(
                    net = models.DenseNet(
                        headsize = 96,
                        bodysize = 32,
                        tailsize = 1,
                        layers = 2,
                        dropout = 0.2,
                        bias = True
                    )
                ),
                
                lin = models.DenseNet(
                    headsize = 96,
                    bodysize = 32,
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                
                weight = 1
                
            ),
            
            models.DistillationBlock(
            
                # 14 -> 7
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(96, 64, 3, padding=1, groups=32),
                    torch.nn.LeakyReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.BatchNorm2d(64)
                ),
                
                atn = models.AttentionPool(
                    net = models.DenseNet(
                        headsize = 64,
                        bodysize = 32,
                        tailsize = 1,
                        layers = 2,
                        dropout = 0.2,
                        bias = True
                    )
                ),
                
                lin = models.DenseNet(
                    headsize = 64,
                    bodysize = 32,
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                
                weight = 0.3
            
            ),
            
            models.DistillationBlock(
            
                # 7 -> 4
                cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                    torch.nn.LeakyReLU(),
                    torch.nn.AvgPool2d(3, padding=1, stride=2),
                    torch.nn.BatchNorm2d(32)
                ),
                
                atn = models.AttentionPool(
                    net = models.DenseNet(
                        headsize = 32,
                        bodysize = 16,
                        tailsize = 1,
                        layers = 2,
                        dropout = 0.2,
                        bias = True
                    )
                ),
                
                lin = models.DenseNet(
                    headsize = 32,
                    bodysize = 16,
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                
                weight = 0.10
            
            ),
            
            tail = torch.nn.Sequential(
            
                torch.nn.Conv2d(32, 16, 3, padding=1, groups=16),
                torch.nn.LeakyReLU(),
                torch.nn.AvgPool2d(4),
                torch.nn.BatchNorm2d(16),
            
                models.Reshape(16),
                models.DenseNet(
                    headsize = 16,
                    bodysize = 32,
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.1,
                    bias = True
                )
            ),
                
            weight = 0.01
        )

import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
    
        initial_channels = 128 * 3
    
        return torch.nn.Sequential( # Parameter count: 6474
            
            # 28 -> 28
            torch.nn.Conv2d(channels, initial_channels, 3, padding=1, groups=channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(initial_channels),
            
            # 28 -> 14
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = initial_channels,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(3, stride=2, padding=1),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                )
            ),
            
            torch.nn.BatchNorm2d(32),
            models.AssertShape(32, 14, 14)
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(3, stride=2, padding=1),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 16,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                )
            ),
            
            torch.nn.BatchNorm2d(16),
            models.AssertShape(16, 7, 7)
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 16,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(3, stride=2, padding=1),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = 8,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                )
            ),
            
            torch.nn.BatchNorm2d(8),
            models.AssertShape(8, 4, 4)
            
            # 4 -> 1
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 8,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(4, stride=1, padding=0),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 16,
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.1,
                    bias = True
                )
            ),
            
            models.Reshape(classes)
        )

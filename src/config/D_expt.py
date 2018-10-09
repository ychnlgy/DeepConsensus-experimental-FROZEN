import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
    
        core_channels = 64
        initial_channels = core_channels * channels
    
        return torch.nn.Sequential( # Parameter count:
            
            # 28 -> 28
            models.ResNet(
                kernelseq = [3],
                headsize = channels,
                bodysize = initial_channels,
                tailsize = initial_channels,
                layers = 4
            ),
            
            # 28 -> 14
            torch.nn.Conv2d(initial_channels, initial_channels, 3, padding=1, groups=initial_channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(initial_channels),
            
            # 14 -> 7
            torch.nn.Conv2d(initial_channels, initial_channels, 3, padding=1, groups=initial_channels),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(initial_channels),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = initial_channels,
                    bodysize = initial_channels*2,
                    tailsize = initial_channels*2,
                    layers = 1,
                    dropout = 0.2,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(3, stride=2, padding=1),
                summarizer = models.DenseNet(
                    headsize = initial_channels*2,
                    bodysize = core_channels,
                    tailsize = core_channels//2,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                )
            ),
            
            # 4 -> 1
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = core_channels//4,
                    bodysize = core_channels,
                    tailsize = core_channels,
                    layers = 1,
                    dropout = 0.2,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(4),
                summarizer = models.DenseNet(
                    headsize = core_channels,
                    bodysize = core_channels//2,
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                )
            ),
            
            models.Reshape(classes)
        )

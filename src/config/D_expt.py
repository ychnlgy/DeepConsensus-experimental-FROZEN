import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
    
        initial_channels = 256 * 3
    
        return torch.nn.Sequential( # Parameter count: 79650
            
            # 28 -> 28
            models.ResNet(
                kernelseq = [3],
                headsize = channels,
                bodysize = initial_channels,
                tailsize = initial_channels,
                layers = 8
            ),
            
            # 28 -> 14
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = initial_channels,
                    bodysize = 512,
                    tailsize = 256,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(2),
                summarizer = models.DenseNet(
                    headsize = 256,
                    bodysize = 256,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.2,
                    bias = True
                )
            ),
            
            torch.nn.BatchNorm2d(128),
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 256,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(2),
                summarizer = models.DenseNet(
                    headsize = 256,
                    bodysize = 256,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                )
            ),
            
            torch.nn.BatchNorm2d(128),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 256,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(3, padding=1, stride=2),
                summarizer = models.DenseNet(
                    headsize = 256,
                    bodysize = 256,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                )
            ),
            
            torch.nn.BatchNorm2d(128),
            
            # 4 -> 1
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 256,
                    layers = 1,
                    dropout = 0.1,
                    bias = True
                ),
                pool = torch.nn.AvgPool2d(4),
                summarizer = models.DenseNet(
                    headsize = 256,
                    bodysize = 64,
                    tailsize = classes,
                    layers = 3,
                    dropout = 0.1,
                    bias = True
                )
            ),
            
            models.Reshape(classes)
        )

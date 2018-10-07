import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count:
            
            # 28 -> 28
            torch.nn.Conv2d(channels, 32, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 28 -> 28
            torch.nn.Conv2d(32, 32, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 28 -> 14
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                kernel = 2,
                stride = 2,
                padding = 0
            ),
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 8,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = 4,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                kernel = 2,
                stride = 2,
                padding = 0
            ),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 4,
                    bodysize = 32,
                    tailsize = 16,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 16,
                    bodysize = 32,
                    tailsize = 4,
                    layers = 2,
                    dropout = 0.1,
                    bias = True
                ),
                kernel = 2,
                stride = 2,
                padding = 0
            ),
            
            # 4 -> 1
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 4,
                    bodysize = 32,
                    tailsize = 16,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 16,
                    bodysize = 32,
                    tailsize = classes,
                    layers = 3,
                    dropout = 0.1,
                    bias = True
                ),
                kernel = 4,
                stride = 1,
                padding = 0
            ),
            
            models.Reshape(len, classes)
        )

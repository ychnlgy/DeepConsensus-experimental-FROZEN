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
            torch.nn.Conv2d(channels, 128, 5, padding=2, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            # 28 -> 14
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                kernel = 5,
                stride = 2,
                padding = 2
            ),
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 16,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                kernel = 5,
                stride = 2,
                padding = 2
            ),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 16,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            # 4 -> 1
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
                    tailsize = classes,
                    layers = 2,
                    dropout = 0.0,
                    bias = True
                ),
                kernel = 4,
                stride = 1,
                padding = 0
            ),
            
            models.Reshape(len, classes)
        )

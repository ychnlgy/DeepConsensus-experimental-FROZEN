import torch

from .Base import Base

import models

class D_expt(Base):

    @staticmethod
    def get_paramid():
        return "expt"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 219882
            
            # 28 -> 28
            torch.nn.Conv2d(channels, 256, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 28 -> 14
            torch.nn.Conv2d(256, 32, 3, padding=1, stride=1),
            torch.nn.AvgPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 14 -> 14
            torch.nn.Conv2d(32, 32, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.2,
                    bias = False
                ),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 16,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.2,
                    bias = False
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(8),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 8,
                    bodysize = 16,
                    tailsize = 16,
                    layers = 1,
                    dropout = 0.2,
                    bias = False
                ),
                summarizer = models.DenseNet(
                    headsize = 16,
                    bodysize = 16,
                    tailsize = 8,
                    layers = 1,
                    dropout = 0.2,
                    bias = False
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(8),
            
            # 4 -> 1
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 8,
                    bodysize = 16,
                    tailsize = 16,
                    layers = 1,
                    dropout = 0.2,
                    bias = False
                ),
                summarizer = models.DenseNet(
                    headsize = 16,
                    bodysize = 16,
                    tailsize = classes,
                    layers = 1,
                    dropout = 0.2,
                    bias = False
                ),
                kernel = 4,
                stride = 1,
                padding = 0
            ),
            
            models.Reshape(len, classes)
        )

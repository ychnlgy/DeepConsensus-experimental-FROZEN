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
            torch.nn.Conv2d(channels, 64, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 28 -> 28
            torch.nn.Conv2d(64, 32, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 28 -> 28
            torch.nn.Conv2d(32, 32, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 16,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(16),
            
            # 4 -> 2
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 16,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 16,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(16),
            
            # 2 -> 1
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 16,
                    bodysize = 64,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.2,
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
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(8),
            
            models.Reshape(len, 8),
            
            models.DenseNet(
                headsize = 8,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.1,
                bias = True
            ),
        )

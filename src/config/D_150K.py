import torch

from .Base import Base

import models

class D_150K(Base):

    @staticmethod
    def get_paramid():
        return "150K"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 149378
            
            torch.nn.Sequential(
                
                # 28 -> 28
                torch.nn.Conv2d(channels, 64, 3, padding=1, stride=1),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(64),
                
                # 28 -> 14
                torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(128),
            
            ),
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.1,
                    bias = False
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.2,
                    bias = False
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 128,
                    tailsize = 16,
                    layers = 2,
                    dropout = 0.2,
                    bias = False
                ),
                summarizer = models.DenseNet(
                    headsize = 16,
                    bodysize = 64,
                    tailsize = 8,
                    layers = 2,
                    dropout = 0.2,
                    bias = False
                ),
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            models.Reshape(len, 8, 16, contiguous=True),
            models.Permute(0, 2, 1), # N, W*H, C
            models.Mean(dim=1), # N, C
            
            models.DenseNet(
                headsize = 8,
                bodysize = 64,
                tailsize = classes,
                layers = 3,
                dropout = 0.2,
                bias = False
            ),
        )

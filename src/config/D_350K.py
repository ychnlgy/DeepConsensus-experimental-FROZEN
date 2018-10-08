import torch

from .Base import Base

import models

class D_350K(Base):

    '''
    
    Highly successfuly for circles vs squares translate and magnify.
    
    '''

    @staticmethod
    def get_paramid():
        return "350K"

    def create_net(self, classes, channels):
        return torch.nn.Sequential( # Parameter count: 344666
            
            # 28 -> 28
            torch.nn.Conv2d(channels, 256, 3, padding=1, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            # 28 -> 14
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 256,
                    bodysize = 256,
                    tailsize = 256,
                    layers = 1,
                    dropout = 0.2,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                channels = 256,
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(256),
            
            # 14 -> 7
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 256,
                    bodysize = 256,
                    tailsize = 256,
                    layers = 1,
                    dropout = 0.2,
                    bias = True
                ),
                summarizer = models.DenseNet(
                    headsize = 128,
                    bodysize = 256,
                    tailsize = 32,
                    layers = 2,
                    dropout = 0.2,
                    bias = True
                ),
                channels = 256,
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(256),
            
            # 7 -> 4
            models.DistillationLayer(
                interpreter = models.DenseNet(
                    headsize = 256,
                    bodysize = 256,
                    tailsize = 256,
                    layers = 1,
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
                channels = 256,
                kernel = 3,
                stride = 2,
                padding = 1
            ),
            
            torch.nn.BatchNorm2d(256),
            
            models.Reshape(len, 256, 16, contiguous=True),
            models.Permute(0, 2, 1), # N, W*H, C
            models.Mean(dim=1), # N, C
            
            models.DenseNet(
                headsize = 256,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2,
                bias = True
            ),
        )

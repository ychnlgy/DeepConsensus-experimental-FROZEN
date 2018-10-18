import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            models.DistillPool(
            
                g = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 1,
                    layers = 2,
                    dropout = 0.2,
                    #activation = models.AbsTanh()
                ),
                
                h = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2,
                    dropout = 0.2,
                    #activation = models.AbsTanh()
                )
                
            ),
            
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
            
        )

import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                # 28 -> 28
                models.DistillLayer(
                    convlayer = models.ResNet(
                        kernelseq = [3, 3],
                        headsize = channels,
                        bodysize = 64,
                        tailsize = 64,
                        layers = 8
                    ),
                    masker = models.DenseNet(
                        headsize = 64,
                        bodysize = 32,
                        tailsize = 1,
                        layers = 2,
                        dropout = 0.2,
                        activation = models.AbsTanh()
                    ),
                    dropout = 0.2,
                    interpreter = models.DenseNet(
                        headsize = 64,
                        bodysize = 128,
                        tailsize = 128,
                        layers = 2,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 128,
                        bodysize = 64,
                        tailsize = 32,
                        layers = 2,
                        dropout = 0.2
                    ),
                ),
            
                # 28 -> 14
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64),
                    ),
                    masker = models.DenseNet(
                        headsize = 64,
                        bodysize = 32,
                        tailsize = 1,
                        layers = 2,
                        dropout = 0.2,
                        activation = models.AbsTanh()
                    ),
                    dropout = 0.2,
                    interpreter = models.DenseNet(
                        headsize = 64,
                        bodysize = 128,
                        tailsize = 128,
                        layers = 2,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 128,
                        bodysize = 64,
                        tailsize = 32,
                        layers = 2,
                        dropout = 0.2
                    ),
                ),
                
            ),
            
            models.DenseNet(
                headsize = 32 + 32,
                bodysize = 128,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )

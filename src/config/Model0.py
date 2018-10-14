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
                        bodysize = 256,
                        tailsize = 256,
                        layers = 8
                    ),
                    dropout = 0.2,
                    interpreter = models.DenseNet(
                        headsize = 256,
                        bodysize = 256,
                        tailsize = 32,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1,
                        dropout = 0.0
                    ),
                ),
            
                # 28 -> 14
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 128, 3, padding=1, groups=32),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(128),
                    ),
                    dropout = 0.2,
                    interpreter = models.DenseNet(
                        headsize = 128,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1,
                        dropout = 0.0
                    ),
                ),
                    
                # 14 -> 7
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 128, 3, padding=1, groups=32),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(128),
                    ),
                    dropout = 0.2,
                    interpreter = models.DenseNet(
                        headsize = 128,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 8,
                        tailsize = 8,
                        layers = 1,
                        dropout = 0.0
                    )
                ),
                
                # 7 -> 4
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, 3, padding=1, groups=32),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64),
                    ),
                    dropout = 0.2,
                    interpreter = models.DenseNet(
                        headsize = 64,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 16,
                        bodysize = 2,
                        tailsize = 2,
                        layers = 1,
                        dropout = 0.0
                    )
                ),
                
            ),
            
            models.DenseNet(
                headsize = 16 + 16 + 8 + 2,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.1
            )
        )

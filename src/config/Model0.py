import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.ResNet(
                kernelseq = [3, 3],
                headsize = channels,
                bodysize = 128,
                tailsize = 128,
                layers = 8
            ),
            
            models.DistillNet(
                
                # 28 -> 28
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(128, 64, 3, padding=1, groups=64),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    interpreter = models.DenseNet(
                        headsize = 64,
                        bodysize = 64,
                        tailsize = 128,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 128,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1
                    )
                ),
                
                # 28 -> 14
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    interpreter = models.DenseNet(
                        headsize = 64,
                        bodysize = 64,
                        tailsize = 128,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 128,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1
                    )
                ),
                
                # 14 -> 7
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    interpreter = models.DenseNet(
                        headsize = 32,
                        bodysize = 32,
                        tailsize = 64,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1
                    )
                ),
                
                # 7 -> 4
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(32)
                    ),
                    interpreter = models.DenseNet(
                        headsize = 32,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 8,
                        tailsize = 8,
                        layers = 1
                    )
                ),
                
                
            ),
            
            models.DenseNet(
                headsize = 32 + 32 + 16 + 8,
                bodysize = 128,
                tailsize = classes,
                layers = 2
            )
        )

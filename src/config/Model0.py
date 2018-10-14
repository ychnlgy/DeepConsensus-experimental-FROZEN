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
                tailsize = 64,
                layers = 8
            ),
                
            # 28 -> 14
            models.DistillLayer(
                convlayer = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32),
                    
                    torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
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
                
                # 14 -> 7
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 16, 3, padding=1, groups=16),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(16),
                        
                        torch.nn.Conv2d(16, 16, 3, padding=1, groups=16),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(16)
                    ),
                    interpreter = models.DenseNet(
                        headsize = 16,
                        bodysize = 16,
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
                
                # 7 -> 4
                models.DistillLayer(
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(16, 8, 3, padding=1, groups=8),
                        torch.nn.AvgPool2d(3, padding=1, stride=2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(8),
                        
                        torch.nn.Conv2d(8, 8, 3, padding=1, groups=8),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(8)
                    ),
                    interpreter = models.DenseNet(
                        headsize = 8,
                        bodysize = 8,
                        tailsize = 16,
                        layers = 1,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 16,
                        bodysize = 2,
                        tailsize = 2,
                        layers = 1
                    )
                ),
                
                
            ),
            
            models.DenseNet(
                headsize = 16 + 8 + 2,
                bodysize = 64,
                tailsize = classes,
                layers = 2
            )
        )

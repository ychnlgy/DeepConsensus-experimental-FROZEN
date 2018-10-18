import torch

import models

from .Base import Base

LAYERS = 8

class Cnn(torch.nn.Module):
    
    def __init__(self, channels, classes):
        super(Cnn, self).__init__()
        self.nets = torch.nn.ModuleList(self.create_nets(channels, classes))
    
    def forward(self, X):
        for net in self.nets:
            X = net(X)
            yield X
    
    def create_nets(self, channels, classes):
        return [
            models.ResNet(
                kernelseq = [3, 3],
                headsize = channels,
                bodysize = 64,
                tailsize = 64,
                layers = LAYERS
            ),
            
            torch.nn.Sequential(
            
                # 28 -> 14
                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                torch.nn.MaxPool2d(2),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(64),
            
                # 14 -> 14
                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(64),
                
#                # 14 -> 7
#                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
#                torch.nn.MaxPool2d(2),
#                torch.nn.LeakyReLU(),
#                torch.nn.BatchNorm2d(64),
#                
#                # 7 -> 4
#                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
#                torch.nn.AvgPool2d(3, padding=1, stride=2),
#                torch.nn.LeakyReLU(),
#                torch.nn.BatchNorm2d(64),
#                
#                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
#                torch.nn.LeakyReLU(),
#                torch.nn.BatchNorm2d(64),
#                
#                # 4 -> 1
#                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
#                torch.nn.AvgPool2d(4),
#                torch.nn.LeakyReLU(),
#                torch.nn.BatchNorm2d(64),
#                
#                models.Reshape(64),
#                
#                models.DenseNet(
#                    headsize = 64,
#                    bodysize = 32,
#                    tailsize = classes,
#                    layers = 2,
#                    dropout = 0.2
#                )
            ),
        ]

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                iternet = Cnn(channels, classes),
            
                pools = [
                
                    models.DistillPool(
                    
                        f = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 64,
                            layers = 2,
                            dropout = 0.2,
                            activation = torch.nn.Sigmoid()
                        ),
                    
                        g = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = torch.nn.Sigmoid()
                        ),
                        
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 128,
                            tailsize = 32,
                            layers = 2,
                            dropout = 0.2
                        )
                        
                    ),
                    
                    models.DistillPool(
                    
                        f = models.DenseNet(
                            headsize = 96,
                            bodysize = 32,
                            tailsize = 64,
                            layers = 2,
                            dropout = 0.2,
                            activation = torch.nn.Sigmoid()
                        ),
                        
                        g = models.DenseNet(
                            headsize = 96,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = torch.nn.Sigmoid()
                        ),
                        
                        h = models.DenseNet(
                            headsize = 64,
                            bodysize = 128,
                            tailsize = 64,
                            layers = 2,
                            dropout = 0.2
                        )
                        
                    )
                ],
                
            ),
            
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
            
        )

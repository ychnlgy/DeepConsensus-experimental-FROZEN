import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                # 28 -> 14
                models.DistillLayer(
                
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(channels, 64, 3, padding=1),
                        torch.nn.MaxPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(64)
                    ),
                    
                    counter = torch.nn.Sequential(
                        torch.nn.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1
                        ),
                        torch.nn.Sigmoid()
                    ),
                    
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 32,
                        tailsize = 16,
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
                    
                    counter = torch.nn.Sequential(
                        torch.nn.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1
                        ),
                        torch.nn.Sigmoid()
                    ),
                    
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 32,
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
                    
                    counter = torch.nn.Sequential(
                        torch.nn.DenseNet(
                            headsize = 32,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1
                        ),
                        torch.nn.Sigmoid()
                    ),
                    
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 32,
                        tailsize = 16,
                        layers = 1
                    )
                    
                ),
                
                # 4 -> 2
                models.DistillLayer(
                
                    convlayer = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 16, 3, padding=1, groups=32),
                        torch.nn.AvgPool2d(2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm2d(16)
                    ),
                    
                    counter = torch.nn.Sequential(
                        torch.nn.DenseNet(
                            headsize = 16,
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1
                        ),
                        torch.nn.Sigmoid()
                    ),
                    
                    summarizer = models.DenseNet(
                        headsize = 32,
                        bodysize = 32,
                        tailsize = 16,
                        layers = 1
                    )
                    
                )
            ),
            
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2
            )
            
        )

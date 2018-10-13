import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.ResNet(
                kernelseq = [3],
                headsize = channels,
                bodysize = 32,
                tailsize = 32,
                layers = 8
            ),
            
            # 28 -> 14
            torch.nn.Conv2d(32, 64, 3, padding=1, groups=32),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.2
                ),
                pooler = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1, stride=2),
                    torch.nn.LeakyReLU()
                ),
                summarizer = models.DenseNet(
                    headsize = 128,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.2
                )
            ),
            
            torch.nn.BatchNorm2d(32),
            
            # 7 -> 4
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 1,
                    dropout = 0.2
                ),
                pooler = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1, stride=2),
                    torch.nn.LeakyReLU()
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 16,
                    tailsize = 16,
                    layers = 1,
                    dropout = 0.1
                )
            ),
            
            torch.nn.BatchNorm2d(16),
            
            # 4 -> 1
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 16,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.1
                ),
                pooler = torch.nn.AvgPool2d(4),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = classes,
                    layers = 1,
                    dropout = 0.1
                )
            ),
            
            models.Reshape(classes)
            
#            # 28 -> 14
#            torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
#            torch.nn.MaxPool2d(2),
#            torch.nn.LeakyReLU(),
#            torch.nn.BatchNorm2d(32),
#            
#            # 14 -> 7
#            torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
#            torch.nn.MaxPool2d(2),
#            torch.nn.LeakyReLU(),
#            torch.nn.BatchNorm2d(32),
#            
#            # 7 -> 4
#            torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
#            torch.nn.AvgPool2d(3, padding=1, stride=2),
#            torch.nn.LeakyReLU(),
#            torch.nn.BatchNorm2d(32),
#            
#            # 4 -> 1
#            torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
#            torch.nn.AvgPool2d(4),
#            torch.nn.LeakyReLU(),
#            torch.nn.BatchNorm2d(32),
#            
#            models.Reshape(32),
#            
#            models.DenseNet(
#                headsize = 32,
#                bodysize = 64,
#                tailsize = classes,
#                layers = 2
#            )
        
#            models.DistillNet(
#            
#                # 28 -> 14
#                models.DistillLayer(
#                
#                    convlayer = torch.nn.Sequential(
#                        torch.nn.Conv2d(channels, 64, 3, padding=1),
#                        torch.nn.MaxPool2d(2),
#                        torch.nn.LeakyReLU(),
#                        torch.nn.BatchNorm2d(64)
#                    ),
#                    
#                    counter = torch.nn.Sequential(
#                        models.DenseNet(
#                            headsize = 64,
#                            bodysize = 32,
#                            tailsize = 32,
#                            layers = 1
#                        ),
#                        torch.nn.Sigmoid()
#                    ),
#                    
#                    summarizer = models.DenseNet(
#                        headsize = 32,
#                        bodysize = 32,
#                        tailsize = 16,
#                        layers = 1
#                    )
#                    
#                ),
#                
#                # 14 -> 7
#                models.DistillLayer(
#                
#                    convlayer = torch.nn.Sequential(
#                        torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
#                        torch.nn.MaxPool2d(2),
#                        torch.nn.LeakyReLU(),
#                        torch.nn.BatchNorm2d(32)
#                    ),
#                    
#                    counter = torch.nn.Sequential(
#                        models.DenseNet(
#                            headsize = 32,
#                            bodysize = 32,
#                            tailsize = 32,
#                            layers = 1
#                        ),
#                        torch.nn.Sigmoid()
#                    ),
#                    
#                    summarizer = models.DenseNet(
#                        headsize = 32,
#                        bodysize = 32,
#                        tailsize = 16,
#                        layers = 1
#                    )
#                    
#                ),
#                
#                # 7 -> 4
#                models.DistillLayer(
#                
#                    convlayer = torch.nn.Sequential(
#                        torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
#                        torch.nn.AvgPool2d(3, padding=1, stride=2),
#                        torch.nn.LeakyReLU(),
#                        torch.nn.BatchNorm2d(32)
#                    ),
#                    
#                    counter = torch.nn.Sequential(
#                        models.DenseNet(
#                            headsize = 32,
#                            bodysize = 32,
#                            tailsize = 32,
#                            layers = 1
#                        ),
#                        torch.nn.Sigmoid()
#                    ),
#                    
#                    summarizer = models.DenseNet(
#                        headsize = 32,
#                        bodysize = 32,
#                        tailsize = 16,
#                        layers = 1
#                    )
#                    
#                ),
#                
#                # 4 -> 2
#                models.DistillLayer(
#                
#                    convlayer = torch.nn.Sequential(
#                        torch.nn.Conv2d(32, 16, 3, padding=1, groups=16),
#                        torch.nn.AvgPool2d(2),
#                        torch.nn.LeakyReLU(),
#                        torch.nn.BatchNorm2d(16)
#                    ),
#                    
#                    counter = torch.nn.Sequential(
#                        models.DenseNet(
#                            headsize = 16,
#                            bodysize = 32,
#                            tailsize = 32,
#                            layers = 1
#                        ),
#                        torch.nn.Sigmoid()
#                    ),
#                    
#                    summarizer = models.DenseNet(
#                        headsize = 32,
#                        bodysize = 32,
#                        tailsize = 16,
#                        layers = 1
#                    )
#                    
#                )
#            ),
#            
#            models.DenseNet(
#                headsize = 64,
#                bodysize = 32,
#                tailsize = classes,
#                layers = 2
#            )
            
        )

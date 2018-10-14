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
            
#            # 28 -> 14
#            torch.nn.Conv2d(128, 256, 3, padding=1, stride=1, groups=128),
#            torch.nn.MaxPool2d(2),
#            torch.nn.LeakyReLU(),
#            torch.nn.BatchNorm2d(256),
#            
#            torch.nn.Conv2d(256, 128, 3, padding=1, groups=128),
#            torch.nn.LeakyReLU(),
#            torch.nn.BatchNorm2d(128),
            
            # 28 -> 14
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 128,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.0
                ),
                pooler = torch.nn.Sequential(
                    torch.nn.AvgPool2d(2),
                    #torch.nn.Conv2d(256, 256, 3, padding=1, stride=1, groups=256),
                    #torch.nn.LeakyReLU(),
                    #torch.nn.BatchNorm2d(256),
                    #torch.nn.Conv2d(256, 128, 3, padding=1, groups=128),
                    #torch.nn.LeakyReLU()
                ),
                summarizer = models.DenseNet(
                    headsize = 128,
                    bodysize = 128,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.0
                )
            ),
            
            torch.nn.BatchNorm2d(128),
            
            # 14 -> 7
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 128,
                    bodysize = 128,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.0
                ),
                pooler = torch.nn.Sequential(
                    torch.nn.AvgPool2d(2),
                    #torch.nn.Conv2d(128, 128, 3, padding=1, stride=1, groups=128),
                    #torch.nn.LeakyReLU(),
                    #torch.nn.BatchNorm2d(128),
                    #torch.nn.Conv2d(128, 64, 3, padding=1, groups=64),
                    #torch.nn.LeakyReLU(),
                ),
                summarizer = models.DenseNet(
                    headsize = 128,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 1,
                    dropout = 0.0
                )
            ),
            
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 1,
                    dropout = 0.0
                ),
                pooler = torch.nn.Sequential(
                    torch.nn.AvgPool2d(3, padding=1, stride=2),
                    #torch.nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=64),
                    #torch.nn.LeakyReLU(),
                    #torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                    #torch.nn.LeakyReLU(),
                ),
                summarizer = models.DenseNet(
                    headsize = 64,
                    bodysize = 64,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.0
                )
            ),
            
            torch.nn.BatchNorm2d(32),
            
            # 4 -> 1
            models.DistillLayer(
                interpreter = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    dropout = 0.0
                ),
                pooler = torch.nn.Sequential(
                    torch.nn.AvgPool2d(4),
                    #torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                    #torch.nn.LeakyReLU()
                ),
                summarizer = models.DenseNet(
                    headsize = 32,
                    bodysize = 32,
                    tailsize = classes,
                    layers = 1,
                    dropout = 0.0
                )
            ),
            
            models.Reshape(classes)
            
        )

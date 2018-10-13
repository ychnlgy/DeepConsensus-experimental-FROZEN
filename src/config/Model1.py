import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.ResNet(
                kernelseq = [3],
                headsize = channels,
                bodysize = 128,
                tailsize = 128,
                layers = 8
            ),
            
            # 28 -> 14
            
            # interpreter
            torch.nn.Conv2d(128, 256, 3, padding=1, groups=128),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            
            torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, 3, padding=1, groups=256),
                torch.nn.AvgPool2d(2),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(256),
                torch.nn.Conv2d(256, 128, 3, padding=1, groups=128),
                torch.nn.LeakyReLU()
            ),
            
            # summarizer
            torch.nn.Conv2d(128, 64, 3, padding=1, groups=64),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            
            # interpreter
            torch.nn.Conv2d(64, 128, 3, padding=1, groups=64),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, 3, padding=1, groups=128),
                torch.nn.AvgPool2d(2),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128, 64, 3, padding=1, groups=64),
                torch.nn.LeakyReLU(),
            ),
            
            # summarizer
            torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 7 -> 4
            
            # interpreter
            torch.nn.Conv2d(32, 64, 3, padding=1, groups=32),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                torch.nn.AvgPool2d(3, padding=1, stride=2),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(64, 32, 3, padding=1, groups=32),
                torch.nn.LeakyReLU(),
            ),
            
            # summarizer
            torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 4 -> 1
            
            # interpreter
            torch.nn.Conv2d(32, 64, 3, padding=1, groups=32),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                torch.nn.AvgPool2d(4),
                torch.nn.LeakyReLU()
            ),
            
            models.Reshape(64),
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.1
            )
            
        )

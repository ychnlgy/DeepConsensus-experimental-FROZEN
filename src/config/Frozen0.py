import torch

import models

from .Base import Base

class Model(Base):

    '''
    
    DO NOT CHANGE THE CODE ANYMORE!
    
    67K parameters
    
    Achieves (random initializations):
    
        MNIST translation +-12
            81.8
            84.7
            81.1
            82.4
            82.1
            76.1
            
        MNIST rotation +-45
            87.7
            86.9
            85.9
            
            vs ResNet + CNN (92K parameters)
            89.6
            89.0
            89.8
        
        MNIST rotation +- 60
            75.0
            75.0
            78.5
            
            vs ResNet + CNN (92K parameters)
            77.7
            79.6
        
    
    '''

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
                    bodysize = 32,
                    tailsize = 1,
                    layers = 2,
                    dropout = 0.2,
                    activation = torch.nn.Sigmoid()
                ),
                
                h = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 128,
                    layers = 1,
                    dropout = 0.2
                )
                
            ),
            
            models.DenseNet(
                headsize = 128,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
            
        )

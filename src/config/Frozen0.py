import torch

import models

from .Base import Base

class Model(Base):

    '''
    
    DO NOT CHANGE THE CODE ANYMORE!
    
    Achieves __ on pure mnist-translation test set after 50 epochs.
    
    '''

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.ResNet(
                kernelseq = [3, 3],
                headsize = channels,
                bodysize = 63,
                tailsize = 64,
                layers = 8
            ),
            
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

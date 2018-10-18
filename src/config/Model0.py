import torch

import models

from .Base import Base

LAYERS = 8

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                iternet = models.ResNet(
                    kernelseq = [3, 3],
                    headsize = channels,
                    bodysize = 64,
                    tailsize = 64,
                    layers = LAYERS
                ),
            
                layers = [
                
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
                            tailsize = 64,
                            layers = 2,
                            dropout = 0.2
                        )
                    ) for i in range(LAYERS)
                    
                ],
                
                encoder = torch.nn.GRU(
                    input_size = 64,
                    hidden_size = 32,
                    num_layers = 2,
                    dropout = 0.0
                ),
                
            ),
            
            models.DenseNet(
                headsize = 32,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
            
        )

import torch

import models

from .Base import Base

LAYERS = 8

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                encoder = torch.nn.GRU(
                    input_size = 64,
                    hidden_size = 128,
                    num_layers = 4,
                    batch_first = True,
                    dropout = 0.2,
                    bidirectional= True
                ),
                
                decoder = torch.nn.GRU(
                    input_size = 256,
                    hidden_size = 128,
                    num_layers = 2,
                    batch_first = True,
                    dropout = 0.2
                ),
            
                layers = [
                    models.DistillLayer(
                        dropout = 0.2,
                        masker = models.DenseNet(
                            headsize = 63,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 63,
                            bodysize = 64,
                            tailsize = 64,
                            layers = 2,
                            dropout = 0.2
                        )
                    ) for i in range(LAYERS)
                ],
                
                iternet = models.ResNet(
                    kernelseq = [3, 3],
                    headsize = channels,
                    bodysize = 63,
                    tailsize = 63,
                    layers = LAYERS
                ),
                
            ),
            
            models.DenseNet(
                headsize = 128,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )

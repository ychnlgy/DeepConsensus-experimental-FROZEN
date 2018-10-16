import torch

import models

from .Base import Base

LAYERS = 8

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                # 28 -> 28
                *[models.DistillLayer(
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ) for i in range(LAYERS)],
                
                iternet = models.ResNet(
                    kernelseq = [3, 3],
                    headsize = channels,
                    bodysize = 63,
                    tailsize = 63,
                    layers = LAYERS
                ),
                
            ),
            
            models.DenseNet(
                headsize = 16 * 8,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )

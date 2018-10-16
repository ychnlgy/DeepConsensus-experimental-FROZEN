import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                # 28 -> 28
                models.DistillLayer(
                    convlayer = models.ResNet(
                        kernelseq = [3, 3],
                        headsize = channels,
                        bodysize = 128,
                        tailsize = 128,
                        layers = 8
                    ),
                    dropout = 0.2,
                    masker = models.DenseNet(
                        headsize = 128,
                        bodysize = 64,
                        tailsize = 1,
                        layers = 3,
                        dropout = 0.2,
                        activation = models.AbsTanh()
                    ),
                    interpreter = models.DenseNet(
                        headsize = 128,
                        bodysize = 256,
                        tailsize = 256,
                        layers = 3,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 256,
                        bodysize = 128,
                        tailsize = 128,
                        layers = 2, # deactivated
                        dropout = 0.05
                    ),
                ),
                
            ),
            
            models.DenseNet(
                headsize = 128,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.05
            )
        )

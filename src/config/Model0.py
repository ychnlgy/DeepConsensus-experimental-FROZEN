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
                        bodysize = 32,
                        tailsize = 1,
                        layers = 2,
                        dropout = 0.2,
                        activation = torch.nn.Tanh()
                    ),
                    selector = models.DenseNet(
                        headsize = 128,
                        bodysize = 64,
                        tailsize = 128,
                        layers = 2,
                        dropout = 0.2,
                        activation = torch.nn.Tanh()
                    ),
                    interpreter = models.DenseNet(
                        headsize = 128,
                        bodysize = 96,
                        tailsize = 64,
                        layers = 2,
                        dropout = 0.2
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 0, # deactivated
                        dropout = 0.0
                    ),
                ),
                
            ),
            
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.0
            )
        )

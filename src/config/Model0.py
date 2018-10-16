import torch

import models

from .Base import Base

LAYERS = 8

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                # 1
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 4,
                        tailsize = 4,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 2
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 4,
                        tailsize = 4,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 3
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 8,
                        tailsize = 8,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 4
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 8,
                        tailsize = 8,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 5
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 6
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 16,
                        tailsize = 16,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 7
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                # 8
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
                    ),
                    summarizer = models.DenseNet(
                        headsize = 64,
                        bodysize = 32,
                        tailsize = 32,
                        layers = 1, # deactivated
                        dropout = 0.2
                    ),
                ),
                
                iternet = models.ResNet(
                    kernelseq = [3, 3],
                    headsize = channels,
                    bodysize = 63,
                    tailsize = 63,
                    layers = LAYERS
                ),
                
            ),
            
            models.DenseNet(
                headsize = 4*2 + 8*2 + 16*2 + 32*2,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )

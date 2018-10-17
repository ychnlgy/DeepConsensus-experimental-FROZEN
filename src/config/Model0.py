import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
#                encoder = torch.nn.GRU(
#                    input_size = 32,
#                    hidden_size = 64,
#                    num_layers = 2,
#                    dropout = 0.2
#                ),
            
                layers = [
                
                    # 28 -> 28
                    models.DistillLayer(
                        convlayer = models.ResNet(
                            kernelseq = [3, 3],
                            headsize = channels,
                            bodysize = 63,
                            tailsize = 64,
                            layers = 8
                        ),
                        dropout = 0.0,
                        masker = models.DenseNet(
                            headsize = 64,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 64,
                            bodysize = 128,
                            tailsize = 128,
                            layers = 1,
                            dropout = 0.2
                        )
                    ),
                    
#                    # 28 -> 14
#                    models.DistillLayer(
#                        convlayer = torch.nn.Sequential(
#                            torch.nn.Conv2d(63, 32, 3, padding=1, groups=32),
#                            torch.nn.MaxPool2d(2),
#                            torch.nn.LeakyReLU(),
#                            torch.nn.BatchNorm2d(32)
#                        ),
#                        dropout = 0.0,
#                        masker = models.DenseNet(
#                            headsize = 32,
#                            bodysize = 16,
#                            tailsize = 1,
#                            layers = 2,
#                            dropout = 0.2,
#                            activation = models.AbsTanh()
#                        ),
#                        interpreter = models.DenseNet(
#                            headsize = 32,
#                            bodysize = 32,
#                            tailsize = 32,
#                            layers = 1,
#                            dropout = 0.0
#                        )
#                    ),
#                    
#                    # 14 -> 14
#                    models.DistillLayer(
#                        convlayer = torch.nn.Sequential(
#                            torch.nn.Conv2d(32, 32, 3, padding=1, groups=32),
#                            torch.nn.LeakyReLU(),
#                            torch.nn.BatchNorm2d(32)
#                        ),
#                        dropout = 0.0,
#                        masker = models.DenseNet(
#                            headsize = 32,
#                            bodysize = 16,
#                            tailsize = 1,
#                            layers = 2,
#                            dropout = 0.2,
#                            activation = models.AbsTanh()
#                        ),
#                        interpreter = models.DenseNet(
#                            headsize = 32,
#                            bodysize = 32,
#                            tailsize = 32,
#                            layers = 1,
#                            dropout = 0.0
#                        )
#                    ),
                    
                ],
                
            ),
            
            models.DenseNet(
                headsize = 128,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
            
        )

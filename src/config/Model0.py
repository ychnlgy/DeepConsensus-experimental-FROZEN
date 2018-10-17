import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                encoder = torch.nn.GRU(
                    input_size = 32,
                    hidden_size = 64,
                    num_layers = 2,
                    batch_first = True,
                    dropout = 0.2,
                    bidirectional= True
                ),
                
                decoder = torch.nn.GRU(
                    input_size = 128,
                    hidden_size = 32,
                    num_layers = 1,
                    batch_first = True
                ),
            
                layers = [
                
                    # 28 -> 28
                    models.DistillLayer(
                        convlayer = models.ResNet(
                            kernelseq = [3, 3],
                            headsize = channels,
                            bodysize = 123,
                            tailsize = 123,
                            layers = 8
                        ),
                        dropout = 0.2,
                        masker = models.DenseNet(
                            headsize = 123,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 123,
                            bodysize = 64,
                            tailsize = 32,
                            layers = 2,
                            dropout = 0.2
                        )
                    ),
                    
                    # 28 -> 14
                    models.DistillLayer(
                        convlayer = torch.nn.Sequential(
                            torch.nn.Conv2d(123, 123, 3, padding=1, groups=123),
                            torch.nn.MaxPool2d(2),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(123)
                        ),
                        dropout = 0.2,
                        masker = models.DenseNet(
                            headsize = 123,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 123,
                            bodysize = 64,
                            tailsize = 32,
                            layers = 2,
                            dropout = 0.2
                        )
                    ),
                    
                    # 14 -> 14
                    models.DistillLayer(
                        convlayer = torch.nn.Sequential(
                            torch.nn.Conv2d(123, 123, 3, padding=1, groups=123),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(123)
                        ),
                        dropout = 0.2,
                        masker = models.DenseNet(
                            headsize = 123,
                            bodysize = 32,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 123,
                            bodysize = 64,
                            tailsize = 32,
                            layers = 2,
                            dropout = 0.2
                        )
                    ),
                    
                ],
                
            ),
            
            models.DenseNet(
                headsize = 32,
                bodysize = 64,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )

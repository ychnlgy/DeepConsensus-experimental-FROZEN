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
                    dropout = 0.2
                ),
            
                layers = [
                
                    # 28 -> 28
                    models.DistillLayer(
                        convlayer = torch.nn.Sequential(
                            torch.nn.Conv2d(channels, 64, 3, padding=1),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(64)
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
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            dropout = 0.0
                        )
                    ),
                    
                    # 28 -> 14
                    models.DistillLayer(
                        convlayer = torch.nn.Sequential(
                            torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                            torch.nn.MaxPool2d(2),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(64)
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
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            dropout = 0.0
                        )
                    ),
                    
                    # 14 -> 14
                    models.DistillLayer(
                        convlayer = torch.nn.Sequential(
                            torch.nn.Conv2d(64, 64, 3, padding=1, groups=64),
                            torch.nn.LeakyReLU(),
                            torch.nn.BatchNorm2d(64)
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
                            bodysize = 32,
                            tailsize = 32,
                            layers = 1,
                            dropout = 0.0
                        )
                    ),
                    
                ],
                
            ),
            
            models.DenseNet(
                headsize = 64,
                bodysize = 32,
                tailsize = classes,
                layers = 2,
                dropout = 0.0
            )
        )

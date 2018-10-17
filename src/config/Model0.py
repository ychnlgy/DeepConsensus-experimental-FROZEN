import torch

import models

from .Base import Base

class Model(Base):

    def create_net(self, channels, classes):
        return torch.nn.Sequential(
        
            models.DistillNet(
            
                encoder = torch.nn.RNN(
                    input_size = 128,
                    hidden_size = 256,
                    num_layers = 2,
                    dropout = 0.2
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
                            bodysize = 256,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 123,
                            bodysize = 256,
                            tailsize = 128,
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
                            bodysize = 256,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 123,
                            bodysize = 256,
                            tailsize = 128,
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
                            bodysize = 256,
                            tailsize = 1,
                            layers = 2,
                            dropout = 0.2,
                            activation = models.AbsTanh()
                        ),
                        interpreter = models.DenseNet(
                            headsize = 123,
                            bodysize = 256,
                            tailsize = 128,
                            layers = 2,
                            dropout = 0.2
                        )
                    ),
                    
                ],
                
            ),
            
            models.DenseNet(
                headsize = 256,
                bodysize = 512,
                tailsize = classes,
                layers = 2,
                dropout = 0.2
            )
        )

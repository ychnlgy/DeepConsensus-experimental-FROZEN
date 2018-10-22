import torch

import models

class Model(models.Savable):

    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            
            # === Convs first ===
            torch.nn.Conv2d(channels, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # === Group <-> Role assignment ===
            
            # 28 -> 14
            models.ChannelTransform(
                net = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2
                )
            ),
            models.Grouper(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            models.ChannelTransform(
                net = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2
                )
            ),
            models.Grouper(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            models.ChannelTransform(
                net = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2
                )
            ),
            models.Grouper(kernel=3, padding=1),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 2
            models.ChannelTransform(
                net = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2
                )
            ),
            models.Grouper(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 2 -> 1
            models.ChannelTransform(
                net = models.DenseNet(
                    headsize = 64,
                    bodysize = 128,
                    tailsize = 64,
                    layers = 2
                )
            ),
            models.Grouper(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            models.Reshape(64),
            
            models.Classifier(64, classes)
        )
    
    def forward(self, X):
        return self.net(X)
import torch

import models

class Model(models.Savable):

    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 28 -> 14
            models.InteractionObserver(),
            models.ChannelClassifier(hiddensize=9, classes=16),
            models.ChannelTransform(
                headsize = 16,
                bodysize = 32,
                tailsize = 32,
                layers = 1,
                #dropout = 0.2
            ),
            models.SoftmaxCombine(),

            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 14 -> 7
            models.InteractionObserver(),
            models.ChannelClassifier(hiddensize=9, classes=16),
            models.ChannelTransform(
                headsize = 16,
                bodysize = 32,
                tailsize = 32,
                layers = 1,
                #dropout = 0.2
            ),
            models.SoftmaxCombine(),

            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 7 -> 4
            models.InteractionObserver(),
            models.ChannelClassifier(hiddensize=9, classes=16),
            models.ChannelTransform(
                headsize = 16,
                bodysize = 32,
                tailsize = 32,
                layers = 1,
                #dropout = 0.2
            ),
            models.SoftmaxCombine(kernel=3, padding=1, stride=2),

            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 4 -> 2
            models.InteractionObserver(),
            models.ChannelClassifier(hiddensize=9, classes=16),
            models.ChannelTransform(
                headsize = 16,
                bodysize = 32,
                tailsize = 32,
                layers = 1,
                #dropout = 0.2
            ),
            models.SoftmaxCombine(),

            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            
            # 2 -> 1
            models.InteractionObserver(),
            models.ChannelClassifier(hiddensize=9, classes=16),
            models.ChannelTransform(
                headsize = 16,
                bodysize = 32,
                tailsize = 32,
                layers = 1,
                #dropout = 0.2
            ),
            models.SoftmaxCombine(),
            
            models.Reshape(32),
            models.Classifier(hiddensize=32, classes=classes)
        )
    
    def forward(self, X):
        return self.net(X)

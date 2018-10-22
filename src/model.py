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
            models.Grouper(),
            models.ChannelClassifier(
                observer = models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 9,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 1,
                    #dropout = 0.2
                ),
                classifier = models.Classifier(
                    hiddensize = 64,
                    classes = 64
                )
            ),
            
            # 14 -> 7
            models.Grouper(),
            models.ChannelClassifier(
                observer = models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 9,
                    bodysize = 64,
                    tailsize = 64,
                    layers = 1,
                    #dropout = 0.2
                ),
                classifier = models.Classifier(
                    hiddensize = 64,
                    classes = 64
                )
            ),
            
            # 7 -> 4
            models.Grouper(),
            models.ChannelClassifier(
                observer = models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 9,
                    bodysize = 32,
                    tailsize = 32,
                    layers = 1,
                    #dropout = 0.2
                ),
                classifier = models.Classifier(
                    hiddensize = 32,
                    classes = 32
                )
            ),
            
            # 4 -> 2
            models.Grouper(),
            models.ChannelClassifier(
                observer = models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 9,
                    bodysize = 16,
                    tailsize = 16,
                    layers = 1,
                    #dropout = 0.2
                ),
                classifier = models.Classifier(
                    hiddensize = 16,
                    classes = 16
                )
            ),
            
            models.Reshape(16*4),
            
            models.Classifier(64, classes)
        )
    
    def forward(self, X):
        return self.net(X)

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
            
            # 28 -> 14
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 14 -> 7
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # === Group <-> Role assignment ===
            
#            # 28 -> 14
#            models.Grouper(),
#            models.ChannelClassifier(
#                observer = models.InteractionObserver(),
#                net = models.DenseNet(
#                    headsize = 9,
#                    bodysize = 32,
#                    tailsize = 32,
#                    layers = 1,
#                    #dropout = 0.2
#                ),
#                classifier = models.Classifier(
#                    hiddensize = 32,
#                    classes = 16
#                )
#            ),
#            
#            # 14 -> 7
#            models.Grouper(),
#            models.ChannelClassifier(
#                observer = models.InteractionObserver(),
#                net = models.DenseNet(
#                    headsize = 9,
#                    bodysize = 32,
#                    tailsize = 32,
#                    layers = 1,
#                    #dropout = 0.2
#                ),
#                classifier = models.Classifier(
#                    hiddensize = 32,
#                    classes = 16
#                )
#            ),
            
            # 7 -> 4
            models.Grouper(kernel=3, padding=1),
            models.ChannelClassifier(
                observer = torch.nn.AvgPool2d(3, padding=1, stride=1),#models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 64,
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
                observer = torch.nn.AvgPool2d(3, padding=1, stride=1),#models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 32,
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
            
            # 2 -> 1
            models.Grouper(),
            models.ChannelClassifier(
                observer = torch.nn.AvgPool2d(3, padding=1, stride=1),#models.InteractionObserver(),
                net = models.DenseNet(
                    headsize = 32,
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
            
            models.Reshape(32),
            
            models.Classifier(32, classes)
        )
    
    def forward(self, X):
        return self.net(X)

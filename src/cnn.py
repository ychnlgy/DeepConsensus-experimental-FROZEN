import torch

import models

class Cnn(models.Savable):

    def __init__(self, channels, classes, imagesize, *args, **kwargs):
        super(Cnn, self).__init__()
        
        if imagesize == (32, 32):
            firstpool = torch.nn.Sequential()
        elif imagesize == (64, 64):
            firstpool = torch.nn.MaxPool2d(2)
        else:
            raise AssertionError
        
        self.layers = torch.nn.ModuleList([
            
            torch.nn.Sequential(
                torch.nn.Conv2d(channels, 32, 5, padding=2),
                firstpool,
                torch.nn.BatchNorm2d(32),
                torch.nn.LeakyReLU(),

                torch.nn.Conv2d(32, 32, 3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.LeakyReLU(),

                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(),
                
                # 32 -> 16
                torch.nn.MaxPool2d(2),

                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(),
                
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(),
                
            ),
            
            
            torch.nn.Sequential(
                # 16 -> 8
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(128, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(),

                torch.nn.Conv2d(128, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(),
            ),
            
            torch.nn.Sequential(
            
                # 8 -> 4
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(256, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(),
                
                torch.nn.Conv2d(256, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(),
            ),
            
        ])
        
        self.net = torch.nn.Sequential(
            torch.nn.AvgPool2d(4),
            models.Reshape(256),
            
            torch.nn.Linear(256, 1024),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            
            torch.nn.Linear(1024, classes)
        )
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return self.net(X)

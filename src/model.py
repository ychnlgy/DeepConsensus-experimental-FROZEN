import torch

import models

class Model(torch.nn.Module):

    def __init__(self, channels, classes):
        super(Model, self).__init__()
        print("OK")
        self.net = torch.nn.Sequential(
            
            # Do alternating convs, gravity pooling
            
            # 64 -> 64
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 64 -> 32
            models.UniqueSquash(),
            models.GravityField(64, 32),
            
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 32 -> 16
            models.UniqueSquash(),
            models.GravityField(32, 16),
            
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 16 -> 8
            models.UniqueSquash(),
            models.GravityField(16, 8),
            
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 8 -> 4
            models.UniqueSquash(),
            models.GravityField(8, 4),
            
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.AvgPool2d(4),
            
            models.Reshape(32),
            torch.nn.Linear(32, classes)
        )
    
    def forward(self, X):
        return self.net(X)

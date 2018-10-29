import torch

import models

class Model(torch.nn.Module):
    
    def __init__(self, channels, classes):
        super(Model, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32)
        )
        
        self.resnet = models.ResNet(
                
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32)
                ),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32)
                ),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32)
                ),
                output = False
            ),
            
            # 32 -> 16
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(64)
                ),
                shortcut = torch.nn.Conv2d(32, 64, 1, stride=2),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(64)
                ),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(64)
                )
            ),
            
            # 16 -> 8
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(128)
                ),
                shortcut = torch.nn.Conv2d(64, 128, 1, stride=2),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(128)
                ),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(128)
                )
            ),
            
            # 8 -> 4
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(256)
                ),
                shortcut = torch.nn.Conv2d(128, 256, 1, stride=2),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(256)
                ),
                output = False
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(256)
                )
            )
        
        )
        
        self.net = torch.nn.Sequential(
            torch.nn.AvgPool2d(4),
            models.Reshape(256),
            torch.nn.Linear(256, classes)
        )
    
    def forward(self, X):
        return self.net(self.resnet(self.conv(X)))

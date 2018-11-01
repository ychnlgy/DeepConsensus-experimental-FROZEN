import torch

import models

class Model(models.Savable):
    
    def __init__(self, channels, classes):
        super(Model, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
        )
        
        self.resnet = models.ResNet(
                
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                ),
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                ),
            ),
            
            # 32 -> 16
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(),
                ),
                shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 1, stride=2),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU()
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(),
                ),
            ),
            
            # 16 -> 8
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                ),
                shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 1, stride=2),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU()
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                ),
            ),
            
            # 8 -> 4
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(),
                ),
                shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, 1, stride=2),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU()
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(),
                ),
            ),
        
        )
        
        self.net = torch.nn.Sequential(
            torch.nn.AvgPool2d(4),
            models.Reshape(256),
            torch.nn.Linear(256, classes)
        )
        
    def forward(self, X):
        return self.net(self.resnet(self.conv(X)))

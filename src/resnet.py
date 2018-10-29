import torch

import models

class Model(torch.nn.Module):
    
    def __init__(self, channels, classes):
        super(Model, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 5, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32)
        )
        
        self.resnet = models.ResNet(
                
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32)
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32)
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(32)
                )
            ),
            
            # 64 -> 32
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(64)
                ),
                shortcut = torch.nn.Conv2d(32, 64, 1, stride=2)
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(64)
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(64)
                )
            ),
            
            # 32 -> 16
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(128)
                ),
                shortcut = torch.nn.Conv2d(64, 128, 1, stride=2)
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(128)
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(128)
                )
            ),
            
            # 16 -> 8
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(256)
                ),
                shortcut = torch.nn.Conv2d(128, 256, 1, stride=2)
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(256)
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(256)
                )
            ),
            
            # 8 -> 4
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 512, 3, padding=1),
                    torch.nn.MaxPool2d(2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(512)
                ),
                shortcut = torch.nn.Conv2d(256, 512, 1, stride=2)
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(512, 512, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(512)
                )
            ),
            
            models.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(512, 512, 3, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(512)
                )
            )
        
        )
        
        self.net = torch.nn.Sequential(
            torch.nn.AvgPool2d(4),
            models.Reshape(512),
            torch.nn.Linear(512, 1024)
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, classes)
        )
    
    def forward(self, X):
        return self.net(self.resnet(self.conv(X)))

import torch

import models

class Model(torch.nn.Module):

    def __init__(self, channels, classes, lamb):
        super(Model, self).__init__()
        
        self.lamb = lamb
        
        self.cnn = torch.nn.Sequential(
        
            # 28 -> 28
            torch.nn.Conv2d(channels, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 28 -> 14
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            # 14 -> 7
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 7 -> 4
            torch.nn.MaxPool2d(3, padding=1, stride=2),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            # 4 -> 1
            torch.nn.Conv2d(64, 64, 4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            
            models.Reshape(64),
        )
        
        self.instance_separator = models.Norm(p=2)
        self.transform = torch.nn.Linear(64, 32)
        self.grouper = models.Classifier(32, classes)
        
        self.group_loss = torch.nn.CrossEntropyLoss()
    
    def calc_loss(self, X, y):
        latent_vecs, group = self._forward(X)
        norms = self.instance_separator(latent_vecs, latent_vecs)
        
        if type(y) is int:
            y = torch.LongTensor([y] * len(X)).to(X.device)
        
        tloss = self.group_loss(group, y) - self.lamb*norms.mean()
        return group, tloss
    
    def forward(self, X):
        return self._forward(X)[1]
    
    def _forward(self, X):
        latent_vecs = self.cnn(X)
        trans = self.transform(latent_vecs)
        group = self.grouper(trans)
        return latent_vecs, group

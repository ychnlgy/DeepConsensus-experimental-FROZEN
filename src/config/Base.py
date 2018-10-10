import torch

import models

class Base(models.Savable):
    
    def __init__(self, channels, classes):
        super(Base, self).__init__()
        self.net = self.create_net(channels, classes)
    
    def forward(self, X):
        return self.net(X)

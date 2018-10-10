import torch

import models

class Model(models.Savable):
    
    def __init__(self, channels, classes):
        super(Model, self).__init__()
        self.net = self.create_net(channels, classes)
    
    def forward(self, X):
        return self.net(X)

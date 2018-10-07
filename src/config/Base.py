import torch

class Base(torch.nn.Module):

    def __init__(self, classes, channels):
        super(Base, self).__init__()
        self.net = self.create_net(classes, channels)
    
    def forward(self, X):
        return self.net(X)
    
    @staticmethod
    def get_paramid():
        raise NotImplementedError
    
    def create_net(self, classes, channels):
        raise NotImplementedError

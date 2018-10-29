import torch

IDENTITY = torch.nn.Sequential()
LEAKY_RELU = torch.nn.LeakyReLU()

class ResBlock(torch.nn.Module):

    def __init__(self, conv, shortcut=IDENTITY, activation=LEAKY_RELU):
        super(ResBlock, self).__init__()
        self.cn = conv
        self.sc = shortcut
        self.ac = activation
    
    def forward(self, X):
        return self.ac(self.cn(X) + self.sc(X))

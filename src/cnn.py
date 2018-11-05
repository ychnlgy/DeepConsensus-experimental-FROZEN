import torch

class Model(torch.nn.Module):

    def __init__(self, channels, classes, imagesize, *args, **kwargs):
        super(Model, self).__init__()
        
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
            
            
            
        ])

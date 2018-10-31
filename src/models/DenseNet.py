import torch

DEFAULT_ACTIVATION = torch.nn.LeakyReLU()

class DenseNet(torch.nn.Module):
    
    def __init__(self, headsize, tailsize=None, bodysize=None, layers=1, dropout=0.0, bias=True, activation=DEFAULT_ACTIVATION, default=DEFAULT_ACTIVATION):
        super(DenseNet, self).__init__()
        self.bias = bias
        
        if tailsize is None:
            tailsize = headsize
        
        if layers == 0:
            self.net = torch.nn.Sequential()
        elif layers == 1:
            self.net = self.create_unit(headsize, tailsize, activation)
        else:
            self.net = torch.nn.Sequential(
                self.create_unit(headsize, bodysize, default, dropout),
                torch.nn.Sequential(*[
                    self.create_unit(bodysize, bodysize, default, dropout)
                    for i in range(layers-2)
                ]),
                self.create_unit(bodysize, tailsize, activation)
            )
        
    def get_net(self):
        return self.net
    
    def forward(self, X):
        return self.net(X)
    
    def create_unit(self, inputsize, outputsize, activation, dropout=0.0):
        return torch.nn.Sequential(
            torch.nn.Linear(inputsize, outputsize, bias=self.bias),
            torch.nn.Dropout(dropout),
            activation
        )

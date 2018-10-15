import torch

DEFAULT_ACTIVATION = torch.nn.LeakyReLU()

class DenseNet(torch.nn.Module):
    
    def __init__(self, headsize, bodysize, tailsize, layers, dropout=0.0, bias=True, activation=DEFAULT_ACTIVATION):
        super(DenseNet, self).__init__()
        assert layers > 0
        self.bias = bias
        
        if layers == 1:
            self.net = self.create_unit(headsize, tailsize, activation)
        else:
            self.net = torch.nn.Sequential(
                self.create_unit(headsize, bodysize, DEFAULT_ACTIVATION, dropout),
                torch.nn.Sequential(*[
                    self.create_unit(bodysize, bodysize, DEFAULT_ACTIVATION, dropout)
                    for i in range(layers-2)
                ]),
                self.create_unit(bodysize, tailsize, activation)
            )
        
    def forward(self, X):
        return self.net(X)
    
    def create_unit(self, inputsize, outputsize, activation, dropout=0.0):
        return torch.nn.Sequential(
            torch.nn.Linear(inputsize, outputsize, bias=self.bias),
            torch.nn.Dropout(dropout),
            activation
        )

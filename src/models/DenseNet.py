import torch

class DenseNet(torch.nn.Module):
    
    def __init__(self, headsize, bodysize, tailsize, layers, dropout, bias):
        super(DenseNet, self).__init__()
        
        assert layers > 0
        
        self.dropout = dropout
        self.bias = bias
        
        if layers == 1:
            self.net = self.create_unit(headsize, tailsize)
        else:
            self.net = torch.nn.Sequential(
                self.create_unit(headsize, bodysize),
                torch.nn.Sequential(*[
                    self.create_unit(bodysize, bodysize)
                    for i in range(layers-2)
                ]),
                torch.nn.Linear(bodysize, tailsize),
                torch.nn.ReLU()
            )
        
    def forward(self, X):
        return self.net(X)
    
    def create_unit(self, inputsize, outputsize):
        return torch.nn.Sequential(
            torch.nn.Linear(inputsize, outputsize, bias=self.bias),
            torch.nn.Dropout(self.dropout),
            torch.nn.ReLU()
        )

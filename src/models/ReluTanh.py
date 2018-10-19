import torch

class ReluTanh(torch.nn.LeakyReLU):
    
    def forward(self, X):
        return super(ReluTanh, self).forward(torch.tanh(X))

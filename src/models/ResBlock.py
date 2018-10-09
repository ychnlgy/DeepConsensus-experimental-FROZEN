import torch

import misc

class ResBlock(torch.nn.Module):
    
    def __init__(self, kernelseq, headsize, bodysize, tailsize):
        super(ResBlock, self).__init__()
        
        if len(kernelseq) == 1:
            net = [self.create_unit(headsize, tailsize, kernelseq[0])]
        else:
            net = [self.create_unit(headsize, bodysize, kernelseq[0])]
            net.extend([
                self.create_unit(bodysize, bodysize, k)
                for k in kernelseq[1:-1]
            ])
            net.append(self.create_unit(bodysize, tailsize, kernelseq[-1]))
        
        self.net = torch.nn.Sequential(*net)
    
    def forward(self, X):
        return self.net(X)
    
    def create_unit(self, c_in, c_out, kernel):
        return torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, kernel, padding=kernel//2, groups=c_in),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(c_out)
        )

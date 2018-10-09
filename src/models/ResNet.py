import torch

from .ResBlock import ResBlock

class ResNet(torch.nn.Module):

    def __init__(self, kernelseq, headsize, bodysize, tailsize, layers):
        super(ResNet, self).__init__()
        
        assert layers > 0
        
        if layers == 1:
            blocks = [ResBlock(kernelseq, headsize, bodysize, tailsize)]
        
        else:
            blocks = [ResBlock(kernelseq, headsize, bodysize, bodysize)]
            blocks.extend([
                ResBlock(kernelseq, bodysize, bodysize, bodysize)
                for _ in range(layers -2)
            ])
            blocks.append(ResBlock(kernelseq, bodysize, bodysize, tailsize))
        
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, X):
        for block in self.blocks[:-1]:
            X = block(X) + X
        return self.blocks[-1](X)

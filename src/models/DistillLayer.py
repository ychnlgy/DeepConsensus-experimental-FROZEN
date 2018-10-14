import torch

import misc

class DistillLayer(torch.nn.Module):

    def __init__(self, convlayer, interpreter, summarizer):
        super(DistillLayer, self).__init__()
        self.convlayer   = convlayer
        self.interpreter = interpreter
        self.summarizer  = summarizer
    
    def forward(self, X):
        convout = self.convlayer(X)
        interpd = self.interpreter(convout.permute(0, 2, 3, 1)) # N, W, H, C
        N, W, H, C = interpd.size()
        interpd = interpd.view(N, W*H, C).mean(dim=1)
        summary = self.summarizer(interpd)
        return convout, summary

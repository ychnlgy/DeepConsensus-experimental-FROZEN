import torch

import misc

class DistillLayer(torch.nn.Module):

    def __init__(self, convlayer, dropout, interpreter, summarizer):
        super(DistillLayer, self).__init__()
        self.convlayer   = convlayer
        self.interpreter = interpreter
        self.summarizer  = summarizer
        self.dropout     = torch.nn.Dropout2d(p=dropout)
    
    def forward(self, X):
        convout = self.convlayer(X)
        convinp = self.dropout(convout)
        interpd = self.interpreter(convinp.permute(0, 2, 3, 1)) # N, W, H, C
        N, W, H, C = interpd.size()
        interpd = interpd.view(N, W*H, C).sum(dim=1)
        summary = self.summarizer(interpd)
        return convout, summary

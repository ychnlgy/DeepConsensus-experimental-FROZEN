import torch

import misc

class DistillLayer(torch.nn.Module):

    def __init__(self, convlayer, masker, dropout, interpreter, summarizer):
        super(DistillLayer, self).__init__()
        self.convlayer   = convlayer
        self.masker      = masker
        self.interpreter = interpreter
        self.summarizer  = summarizer
        self.dropout     = torch.nn.Dropout2d(p=dropout)
    
    def forward(self, X):
        convout = self.convlayer(X)
        convinp = self.dropout(convout)
        convinp = convinp.permute(0, 2, 3, 1) # N, W, H, C
        N, W, H, C = convinp.size()
        maskout = self.masker(convinp).view(N, W*H, 1)
        interpd = self.interpreter(convinp)
        N, W, H, C = interpd.size()
        interpd = interpd.view(N, W*H, C)
        spooled = (maskout * interpd).sum(dim=1)
        summary = self.summarizer(spooled)
        return convout, summary

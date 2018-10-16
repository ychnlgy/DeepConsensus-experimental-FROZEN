import torch

import misc

EMPTY = torch.nn.Sequential()

class DistillLayer(torch.nn.Module):

    def __init__(self, masker, dropout, interpreter, summarizer, convlayer=EMPTY):
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
        convinp = convinp.view(N, W*H, C)
        maskout = self.masker(convinp)
        interpd = self.interpreter(convinp)
        spooled = (maskout * interpd).sum(dim=1)
        summary = self.summarizer(spooled)
        return convout, summary

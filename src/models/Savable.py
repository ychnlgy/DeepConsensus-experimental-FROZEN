import torch

from .util import paramcount

class Savable(torch.nn.Module):

    def save(self, fname):
        torch.save(self.state_dict(), fname)
    
    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location="cpu"))
    
    def paramcount(self):
        return paramcount(self)

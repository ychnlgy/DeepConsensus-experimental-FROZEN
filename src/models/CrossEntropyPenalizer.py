import torch

EMPTY = torch.nn.Parameter(torch.zeros(1))

class CrossEntropyPenalizer(torch.nn.CrossEntropyLoss):

    def forward(self, yh, y):
    
        '''
        
        Calculates cross entropy loss on the incorrect predictions only.
        
        '''
    
        vals, indx = yh.max(dim=1)
        wrong = (indx != y)
        
        yh = yh[wrong]
        y  = y [wrong]
        
        if len(y) == 0:
            return EMPTY
        else:
            return super(CrossEntropyPenalizer, self).forward(yh, y)

import torch

class CrossEntropyPenalizer(torch.nn.CrossEntropyLoss):

    def forward(self, yh, y):
    
        '''
        
        Calculates cross entropy loss on the incorrect predictions only.
        
        '''
    
        vals, indx = yh.max(dim=1)
        wrong = (indx != y)
        return super(CrossEntropyPenalizer, self).forward(yh[wrong], y[wrong])

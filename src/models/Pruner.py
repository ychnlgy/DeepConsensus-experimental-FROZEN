import torch

import misc

from .Counter import Counter
from .DistributionTracker import DistributionTracker

class Pruner(torch.nn.Module):

    '''
    
    Discovers the features that are irrelevant or relevant
    for the classification task, and sets corresponding weights
    equal to 0 and 1 respectively.
    
    Parameters:
        delta - distribution means of difference greater than
                delta*(stdev1 + stdev2) are considered relevant.
        classes - int number of class labels to track
        
    '''

    def __init__(self, delta, classes, patience):
        super(Pruner, self).__init__()
        self.tracked = False
        self.patience = patience
        self.waited = 0
        self.delta = delta
        self.counter = Counter()
        self.tracker = DistributionTracker(classes)
        self.lowest = float("inf")
    
    def forward(self, X, vscore=None, labels=None, usescore=False):
        
        '''
        
        Given:
            X - Tensor of shape (N, C, W, H)
            labels - (optional during evaluation) Tensor of shape (N)
        
        Returns:
            Tensor of shape (N, C, W, H), input with selected
            channels set to zero if deemed not correlated to classification.
            Also returns Tensor of shape (N, C), the summed count of features.
        
        '''
        
        self.setup(X)
        counted = self.counter(X)
        
        if self.training:
        
            if self.tracked and usescore:
                
                if vscore >= self.lowest:
            
                    if self.waited >= self.patience:
                        self.waited = 0
                        self.prune()
                        
                    else:
                        self.waited += 1
                
                else:
                    self.lowest = vscore
                    self.waited = 0
            
            self.tracker.reset()
            self.tracked = False

            assert labels is not None
            self.tracker(counted, labels)
            self.tracked = True
        
        return X, counted*self.weights
    
    def prune(self):
    
        '''
        
        Description:
            Updates weights for selecting which channels to zero out.
        
        '''
        self.weights = self.find_correlations()
        print("Using %d/%d channels" % (self.weights.sum(), self.weights.numel()))
    
    # === PRIVATE ===
    
    def setup(self, X):
        N, C, W, H = X.size()
        # all features are considered at first
        self.weights = torch.ones(C).to(X.device)
        self.setup = misc.util.do_nothing
    
    def find_correlations(self):
        
        '''
        
        Returns:
            ByteTensor of shape (classes, C), mask of which features
            have means that are delta*stdev different
            from their expected value.
        
        '''
        
        local_miu, local_std, global_miu, global_std = self.tracker.stats()
        diff_pos = local_miu - global_miu
        diff_neg = global_miu - local_miu
        dist = global_std * self.delta
        diff_pos[diff_pos <=dist] = 0
        diff_pos[diff_pos > dist] = 1
        diff_neg[diff_neg <=dist] = 0
        diff_neg[diff_neg > dist] = 1
        
        diff_pos = diff_pos.sum(dim=0)
        diff_neg = diff_neg.sum(dim=0)
        return (diff_pos > 0) | (diff_neg > 0)

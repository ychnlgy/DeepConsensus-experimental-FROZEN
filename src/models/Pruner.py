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

    def __init__(self, delta, classes, prune_rest):
        super(Pruner, self).__init__()
        self.tracked = False
        self.prune_rest = prune_rest
        self.prune_num  = 0
        self.delta = delta
        self.counter = Counter()
        self.tracker = DistributionTracker(classes)
    
    def forward(self, X, labels=None):
        
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
        
        if not self.training and self.tracked:
            self.prune()
            self.tracked = False
        
        X = self.weights * X #+ (1 - self.weights) * self.miu
        counted = self.counter(X)
        
        if self.training and labels is not None:
            self.tracker(counted, labels)
            self.tracked = True
        
        return X, counted
    
    def prune(self):
    
        '''
        
        Description:
            Updates weights for selecting which channels to zero out.
        
        '''
        
        if self.prune_num % self.prune_rest == 0:
            diff = self.find_correlations()
            diff = self.xor(diff).float()
            self.tracker.reset()
            
            assert diff.size() == self.weights.size()
            newd = (self.weights.sum() - diff.sum()).item()
            self.weights = diff
            
            #assert newd >= 0
            if newd > 0:
                print("Using %d/%d channels" % (self.weights.sum(), self.weights.numel()))
    
        self.prune_num += 1
    
    # === PRIVATE ===
    
    def setup(self, X):
        N, C, W, H = X.size()
        # all features are considered at first
        self.weights = torch.ones(1, C, 1, 1).to(X.device)
        #self.miu = torch.zeros(1, C, 1, 1).to(X.device)
        self.setup = misc.util.do_nothing
    
    def find_correlations(self):
        
        '''
        
        Returns:
            ByteTensor of shape (classes, C), mask of which features
            have means that are delta*(stdev1 + stdev2) different
            from their expected value.
        
        '''
        
        local_miu, local_std, global_miu, global_std = self.tracker.stats()
        #self.miu = global_miu.view(1, -1, 1, 1)
        diff = (global_miu - local_miu).abs()
        dist = (global_std) * self.delta
        diff[diff < dist] = 0
        diff[diff >=dist] = 1
        return diff.byte()
    
    def xor(self, diff):
    
        '''
        
        Given:
            ByteTensor of shape (classes, C)
        
        Returns:
            ByteTensor of shape (1, C, 1, 1), mask representing
            the features that are pertinent for distinguishing
            between the classes.
        
        '''
    
        classes, C = diff.size()
        diff = diff.sum(dim=0)
        xor = (diff > 0)# & (diff < classes)
        xor = xor.view(1, C, 1, 1)
        return xor

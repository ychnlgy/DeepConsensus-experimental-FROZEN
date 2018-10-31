import torch

class NormalInit:
    
    def init_weights(self, sequential):
        assert isinstance(sequential, torch.nn.Sequential)
        target_types = self.get_init_targets()
        for module in sequential:
            if type(module) in target_types:
                torch.nn.init.xavier_uniform(module.weight)
    
    def get_init_targets(self):
        raise NotImplementedError

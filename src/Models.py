import torch

import models, config

class Base(models.Savable):
    
    def __init__(self, paramid, classes, channels):
        super(Base, self).__init__()
        self.net = self.choose_net(paramid, classes, channels)
    
    @staticmethod
    def expected_imagesizes():
        raise NotImplementedError
    
    def choose_net(self, paramid, classes, channels):
        return {
            Net.get_paramid(): Net
            for Net in self.get_nets()
        }[paramid](classes, channels)
    
    def forward(self, X):
        return self.net(X)

class Cnn28or32(Base):

    @staticmethod
    def expected_imagesizes():
        return [
            (28, 28),
            (32, 32)
        ]
    
    def get_nets(self):
        return [
            config.Random,
            config.C_30K,
            config.C_150K,
            config.C_350K,
            config.C_500K
        ]

class DistillationNetwork28or32(Base):

    @staticmethod
    def expected_imagesizes():
        return [
            (28, 28),
            (32, 32)
        ]
    
    def get_nets(self):
        return [
            config.D_30K,
            config.D_150K,
            config.D_350K,
            config.D_450K,
            config.D_500K
        ]

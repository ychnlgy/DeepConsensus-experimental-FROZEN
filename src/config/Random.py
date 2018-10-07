import torch

from .Base import Base

import models

class Random(Base):

    @staticmethod
    def get_paramid():
        return "random"

    def create_net(self, classes, channels):
        return models.RandomLayer(len, classes)

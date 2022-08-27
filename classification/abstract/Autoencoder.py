import abc

from torch import nn


class Autoencoder(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, *input):
        pass

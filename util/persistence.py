import os
import pickle

import torch
from torch import nn

folder = 'data/'

if not os.path.exists(folder):
    os.makedirs(folder)


def save_object(filename: str, data: object):
    with open(folder + filename, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename: str) -> object:
    with open(folder + filename, "rb") as input_file:
        return pickle.load(input_file)


def _get_model_path(model: nn.Module):
    return folder + model.__class__.__name__ + '.model'


def save_model(model: nn.Module):
    torch.save(model.state_dict(), _get_model_path(model))


def load_model(model: nn.Module) -> nn.Module:
    model.load_state_dict(torch.load(_get_model_path(model)))
    return model

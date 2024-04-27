"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

from losses import RangeMAELoss
from utils import set_device
from models import SpectroViT
from save_models import SaveBestModel
from datasets import DatasetThreeChannelSpectrogram
import torch

save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "SpectroViT": SpectroViT,
    },
    "dataset": {
        "DatasetThreeChannelSpectrogram": DatasetThreeChannelSpectrogram
    },
    "optimizer": {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    },
    "loss": {
        "RangeMAELoss": RangeMAELoss
    },
}

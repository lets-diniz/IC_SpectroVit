"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

from losses import RangeMAELoss, RangeMAELossPPMVary
from utils import set_device
from models import SpectroViT
from save_models import SaveBestModel, SaveBestModelState, SaveLossesAndMetrics
from datasets import *
import torch
from scipy import signal,stats

save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "SpectroViT": SpectroViT,
    },
    "dataset": {
        "DatasetThreeChannelSpectrogram": DatasetThreeChannelSpectrogram,
        "DatasetSpgramSyntheticData": DatasetSpgramSyntheticData,
        "DatasetSpgramSyntheticDataOldSTFT": DatasetSpgramSyntheticDataOldSTFT,
        "DatasetSpgramRealData": DatasetSpgramRealData,
        "DatasetSpgramRealDataOldSTFT": DatasetSpgramRealDataOldSTFT,
        "DatasetRealDataSeparateFiles": DatasetRealDataSeparateFiles,
    },
    "optimizer": {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    },
    "loss": {
        "RangeMAELoss": RangeMAELoss,
        "RangeMAELossPPMVary": RangeMAELossPPMVary
    },
    "spgram_window":{
        "window_size": 256,
        "window": signal.windows.hann(256,sym = True)
    },
    "savebest":{
        "SaveBestModelState": SaveBestModelState
    },
    "savelossesandmetrics":{
        "SaveLossesAndMetrics": SaveLossesAndMetrics
    }
}

"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import torch
import torch.nn as nn


class RangeMAELoss(nn.Module):

    def __init__(self):
        super(RangeMAELoss, self).__init__()

    def forward(self, x, y, ppm):
        # defining indexes of boundaries
        gaba_min_ind = torch.argmin(ppm[ppm >= 3.2])
        gaba_max_ind = torch.argmin(ppm[ppm >= 2.8])

        glx_min_ind = torch.argmin(ppm[ppm >= 3.95])
        glx_max_ind = torch.argmin(ppm[ppm >= 3.55])

        # selecting part of arrays pertaining to region of interest
        gaba_x = x[:, gaba_min_ind:gaba_max_ind]
        gaba_y = y[:, gaba_min_ind:gaba_max_ind]

        glx_x = x[:, glx_min_ind:glx_max_ind]
        glx_y = y[:, glx_min_ind:glx_max_ind]

        gaba_mae = torch.abs(gaba_x - gaba_y).mean(dim=1).mean(dim=0)
        glx_mae = torch.abs(glx_x - glx_y).mean(dim=1).mean(dim=0)
        global_mae = torch.abs(x - y).mean(dim=1).mean(dim=0)

        return (gaba_mae * 8 + glx_mae + global_mae) / 10


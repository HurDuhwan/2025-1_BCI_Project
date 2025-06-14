import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
from torch import Tensor
import os
from scipy.signal import butter
from models.cbam import CBAM
from utils.util import butter_bandpass_filter


class VarianceLayer(nn.Module):
    def __init__(self, window_size=1): # the best window size is 1 in paper
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, F, T = x.shape
        x = x.reshape(B, F, T // self.window_size, self.window_size)
        v = x.var(dim=-1, unbiased=False)

        return v

class FeatureExtractor(nn.Module):
    def __init__(self, n_channels, n_bands=6, depth_mul=2, window_size=1, T=1000):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.depth_mul = depth_mul
        # Depthwise spatial conv
        self.depthwise = nn.Conv2d(n_bands, n_bands * depth_mul, (n_channels, 1), groups=n_bands, bias=False)
        # CBAM
        self.cbam = CBAM(n_channels, r=2)

        # VarLayer
        self.var_layer = VarianceLayer(window_size=window_size)
        self.fc = nn.Linear(int(n_bands*depth_mul*T/window_size), 64)

        # Sub-band filtering
        self.fs = 250  # EEG sampling frequency (BCI IV 2a: 250Hz)
        self.band_freqs = [(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]

    def forward(self, x):

        # INPUT DIMENSION: B, 1, C=22, T=1000

        # Feature Extractor Step 1: CBAM
        x = x.permute(0, 2, 1, 3) # [B, C=22, 1, T=1000]
        x = self.cbam(x) # [B, C=22, 1, T=1000]
        x = x.permute(0, 2, 1, 3) # [B, 1, C=22, T=1000]

        # Feature Extractor Step 2: Sub-band Filtering
        x_filtered = []
        x_np = x.detach().cpu().numpy()
        for low, high in self.band_freqs:
            filtered_band = butter_bandpass_filter(x_np, low, high, fs=self.fs, order=5)
            x_filtered.append(filtered_band)
        x_filtered = np.concatenate(x_filtered, axis=1) # sub-bands (B, n_bands, 22, 1000)
        x_filtered = torch.from_numpy(x_filtered).float().to(x.device)

        # Feature Extractor Step 3: Depth-Wise Convolution
        x = self.depthwise(x_filtered) # [B, m=6*d=2, 1, T=1000]
        x = x.squeeze(2)  # (B, m=n_bands=6 * depth_mul=d=2, T=1000)

        # Feature Extractor Step 4: Variation Layer
        x = self.var_layer(x)  # (B, n_bands * depth_mul, T/window_size)

        # Feature Extractor Step 5: FC Layer
        x = x.flatten(1) # (B, n_bands * depth_mul * T / window_size)
        x = self.fc(x) # (B, n_bands * depth_mul * T / window_size)
        # We check if the dimension matched in Table 1 on paper.

        return x

class Classifier(nn.Module):
    def __init__(self, in_dim=64, n_cls=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, int(in_dim/2)),
            nn.Linear(int(in_dim/2), n_cls),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.fc(x)

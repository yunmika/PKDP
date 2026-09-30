#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import torch
import torch.nn as nn
import torch.optim as optim
from log import log, WARNING, ERROR, INFO

MIN_FEATURES_FOR_CONV = 4

class ModelOpts:
    def __init__(self,
                 in_channels=1,
                 out_channels1=64,
                 out_channels2=32,
                 out_channels3=32,
                 fc_layers=2,
                 fc_units=None,
                 out_dim=1,
                 kernel_size=None,
                 dropout_prob=0.2,
                 prior_features=None,
                 **kwargs):
        if fc_units is None:
            fc_units = [128, 64]
        if kernel_size is None:
            kernel_size = [5, 11, 21]
        
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.out_channels3 = out_channels3
        self.fc_layers = fc_layers
        self.fc_units = fc_units
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        self.prior_features = prior_features


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        red = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, red),
            nn.GELU(),
            nn.Linear(red, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class MultiScaleResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=[5, 11, 21]):
        super(MultiScaleResBlock1D, self).__init__()
        k1, k2, k3 = kernel_sizes if len(kernel_sizes) >= 3 else [5, 11, 21]

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k2, padding=k2 // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size=k3, padding=k3 // 2)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return self.act1(out + res)


SequentialCascadedBlock1D = MultiScaleResBlock1D
CascadedResBlock1D = MultiScaleResBlock1D


class EpistasisModule(nn.Module):
    def __init__(self, num_priors, out_dim=64, dropout=0.2, **kwargs):
        super(EpistasisModule, self).__init__()
        self.num_priors = num_priors
        self.num_pairs = (num_priors * (num_priors - 1)) // 2
        triu_i, triu_j = torch.triu_indices(num_priors, num_priors, offset=1)
        self.register_buffer('triu_i', triu_i)
        self.register_buffer('triu_j', triu_j)
        in_dim = num_priors + self.num_pairs
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x_prior):
        x = x_prior.squeeze(1) if x_prior.dim() == 3 else x_prior
        pairs = x[:, self.triu_i] * x[:, self.triu_j]
        feat = torch.cat([x, pairs], dim=-1)
        return self.net(feat)


class PKDP(nn.Module):
    def __init__(self, in_channels=1, out_channels1=64, out_channels2=32, out_channels3=32, 
                 fc_layers=None, fc_units=None, out_dim=1, input_length=3000, 
                 kernel_sizes=None, dropout_prob=0.2, prior_features=None, feature_names=None,
                 prior_dim=64, **kwargs):
        super(PKDP, self).__init__()
        
        self.seq_length = input_length
        self.in_channels = in_channels
        self.prior_features = prior_features
        self.feature_names = feature_names if feature_names is not None else [str(i) for i in range(input_length)]
        
        self.prior_indices = self._get_prior_indices()
        self.num_priors = len(self.prior_indices)
        
        all_indices = set(range(input_length))
        prior_set = set(self.prior_indices)
        self.main_indices = sorted(list(all_indices - prior_set))
        self.num_main = len(self.main_indices)
        
        c1 = out_channels1 or 64
        c2 = out_channels2 or 32
        c3 = out_channels3 or 32
        
        k_sizes = kernel_sizes if kernel_sizes and len(kernel_sizes) >= 3 else [5, 11, 21]
        
        self.conv1 = MultiScaleResBlock1D(in_channels, c1, kernel_sizes=k_sizes)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = MultiScaleResBlock1D(c1, c2, kernel_sizes=k_sizes)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = MultiScaleResBlock1D(c2, c3, kernel_sizes=k_sizes)
        self.pool3 = nn.MaxPool1d(2)
        
        out_len = self.num_main // 8
        self.main_flat_dim = c3 * out_len
        
        if self.num_priors > 0:
            self.prior_net = EpistasisModule(self.num_priors, out_dim=prior_dim, dropout=dropout_prob)
            self.film_gamma = nn.Linear(prior_dim, c3)
            self.film_beta = nn.Linear(prior_dim, c3)
            self.has_priors = True
            total_dim = self.main_flat_dim + prior_dim
        else:
            self.prior_net = None
            self.film_gamma = None
            self.film_beta = None
            self.has_priors = False
            total_dim = self.main_flat_dim
        
        units = fc_units if fc_units else [128, 64]
        fc_list = []
        cur_dim = total_dim
        for u in units:
            fc_list.extend([
                nn.Linear(cur_dim, u),
                nn.BatchNorm1d(u),
                nn.GELU(),
                nn.Dropout(dropout_prob)
            ])
            cur_dim = u
        fc_list.append(nn.Linear(cur_dim, out_dim))
        self.head = nn.Sequential(*fc_list)

    def _get_prior_indices(self):
        if self.prior_features is None:
            return []
        
        if isinstance(self.prior_features, (list, tuple)):
            prior_ids = self.prior_features
        elif isinstance(self.prior_features, str):
            prior_ids = self.prior_features.split()
        else:
            return []
        
        name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        indices = []
        for pid in prior_ids:
            if isinstance(pid, int):
                if 0 <= pid < self.seq_length:
                    indices.append(pid)
            elif str(pid).isdigit():
                idx = int(pid)
                if 0 <= idx < self.seq_length:
                    indices.append(idx)
            else:
                idx = name_to_idx.get(str(pid), -1)
                if idx >= 0:
                    indices.append(idx)
                else:
                    clean_pid = re.sub(r'[^0-9a-zA-Z_]', '_', str(pid))
                    idx = name_to_idx.get(clean_pid, -1)
                    if idx >= 0:
                        indices.append(idx)
        return sorted(list(set(indices)))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x_main = x[:, :, self.main_indices]
        
        f_main = self.pool1(self.conv1(x_main))
        f_main = self.pool2(self.conv2(f_main))
        f_main = self.pool3(self.conv3(f_main))
        
        if self.has_priors:
            x_prior = x[:, :, self.prior_indices]
            f_prior = self.prior_net(x_prior)
            gamma = self.film_gamma(f_prior).unsqueeze(-1)
            beta = self.film_beta(f_prior).unsqueeze(-1)
            f_main_mod = (1.0 + gamma) * f_main + beta
            flat_main = f_main_mod.view(f_main_mod.size(0), -1)
            combined = torch.cat([flat_main, f_prior], dim=-1)
        else:
            flat_main = f_main.view(f_main.size(0), -1)
            combined = flat_main
            
        return self.head(combined)


def create_model(
    opts: ModelOpts = None,
    input_length: int = None,
    in_channels=None,
    out_channels1=None,
    out_channels2=None,
    out_channels3=None,
    fc_layers=None,
    fc_units=None,
    out_dim=None,
    kernel_size=None,
    dropout_prob=None,
    prior_features=None,
    feature_names=None,
    **kwargs
):
    c1 = out_channels1 or (getattr(opts, 'main_channels', [64])[0] if hasattr(opts, 'main_channels') and opts.main_channels else 64)
    c2 = out_channels2 or (getattr(opts, 'main_channels', [64, 32])[1] if hasattr(opts, 'main_channels') and len(opts.main_channels) > 1 else 32)
    c3 = out_channels3 or (getattr(opts, 'main_channels', [64, 32, 32])[2] if hasattr(opts, 'main_channels') and len(opts.main_channels) > 2 else 32)
    
    k_sizes = kernel_size or getattr(opts, 'conv_kernel_size', [5, 11, 21])
    if isinstance(k_sizes, (int, float)):
        k_sizes = [int(k_sizes)] * 3
    elif len(k_sizes) < 3:
        k_sizes = [5, 11, 21]
        
    priors = prior_features if prior_features is not None else getattr(opts, 'prior_features', None)
    units = fc_units or getattr(opts, 'fc_units', [128, 64])
    drp = dropout_prob if dropout_prob is not None else getattr(opts, 'dropout', 0.2)
    in_ch = in_channels or getattr(opts, 'in_channels', 1)
    o_dim = out_dim or getattr(opts, 'out_dim', 1)
    
    if input_length is None:
        raise ValueError("input_length must be specified")
        
    return PKDP(
        in_channels=in_ch,
        out_channels1=c1,
        out_channels2=c2,
        out_channels3=c3,
        fc_units=units,
        out_dim=o_dim,
        input_length=input_length,
        kernel_sizes=k_sizes,
        dropout_prob=drp,
        prior_features=priors,
        feature_names=feature_names
    )


def create_optimizer(model, learning_rate, optimizer_type='AdamW'):
    if optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        log(ERROR, f"Unsupported optimizer type: {optimizer_type}")
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def create_loss_function():
    return nn.MSELoss()
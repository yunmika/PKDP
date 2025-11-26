#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from log import log, WARNING, ERROR, INFO

# Minimum features required for convolution (below this, skip conv and use linear)
MIN_FEATURES_FOR_CONV = 4

class ModelOpts:
    def __init__(self,
                 in_channels=1,
                 out_channels1=64,
                 out_channels2=32,
                 out_channels3=32,
                 prior_channels1=16,
                 prior_channels2=32,
                 prior_channels3=32,
                 fc_layers=3,
                 fc_units=None,
                 out_dim=1,
                 kernel_size=None,
                 prior_kernel_size=None,
                 dropout_prob=0,
                 prior_features=None):
        if fc_units is None:
            fc_units = [128, 64, 32]
        if kernel_size is None:
            kernel_size = [11, 11, 11]
        if prior_kernel_size is None:
            prior_kernel_size = [3, 3, 3]
        
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.out_channels3 = out_channels3
        self.prior_channels1 = prior_channels1
        self.prior_channels2 = prior_channels2
        self.prior_channels3 = prior_channels3
        self.fc_layers = fc_layers
        self.fc_units = fc_units
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.prior_kernel_size = prior_kernel_size
        self.dropout_prob = dropout_prob
        self.prior_features = prior_features

class PKDP(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, 
                 prior_channels1, prior_channels2, prior_channels3,
                 fc_layers, fc_units, out_dim, input_length, 
                 kernel_sizes=None, prior_kernel_sizes=None, 
                 dropout_prob=0.3, prior_features=None, feature_names=None):
        super(PKDP, self).__init__()
        
        self.seq_length = input_length
        self.in_channels = in_channels
        self.prior_features = prior_features
        self.feature_names = feature_names if feature_names is not None else [str(i) for i in range(input_length)]
        
        default_kernel_sizes = [11, 11, 11]
        default_prior_kernel_sizes = [3, 3, 3]
        
        if kernel_sizes is None or len(kernel_sizes) == 0:
            kernel_sizes = default_kernel_sizes.copy()
        elif len(kernel_sizes) == 1:
            kernel_sizes = [kernel_sizes[0]] * 3
        elif len(kernel_sizes) == 2:
            kernel_sizes = [kernel_sizes[0], kernel_sizes[1], kernel_sizes[1]]
        else:
            kernel_sizes = list(kernel_sizes[:3])
        
        if prior_kernel_sizes is None or len(prior_kernel_sizes) == 0:
            prior_kernel_sizes = default_prior_kernel_sizes.copy()
        elif len(prior_kernel_sizes) == 1:
            prior_kernel_sizes = [prior_kernel_sizes[0]] * 3
        elif len(prior_kernel_sizes) == 2:
            prior_kernel_sizes = [prior_kernel_sizes[0], prior_kernel_sizes[1], prior_kernel_sizes[1]]
        else:
            prior_kernel_sizes = list(prior_kernel_sizes[:3])
        
        self.prior_indices = self._get_prior_indices()
        
        num_prior_features = len(self.prior_indices) if self.prior_indices else 0
        num_main_features = input_length - num_prior_features
        
        self.num_prior_features = num_prior_features
        self.num_main_features = num_main_features
        
        all_indices = set(range(input_length))
        prior_set = set(self.prior_indices)
        self.main_indices = sorted(list(all_indices - prior_set))
        
        self.main_use_conv = num_main_features >= MIN_FEATURES_FOR_CONV
        self.prior_use_conv = num_prior_features >= MIN_FEATURES_FOR_CONV
        
        # Build main path
        if num_main_features <= 0:
            self.general_path = None
            self.main_out_channels = 0
            self.main_output_length = 0
        elif num_main_features < MIN_FEATURES_FOR_CONV:
            self.general_path = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_channels * num_main_features, out_channels1),
                nn.ReLU()
            )
            self.main_out_channels = out_channels1
            self.main_output_length = 1
        else:
            self.general_path, self.main_out_channels, self.main_output_length = self._build_conv_path(
                num_main_features, in_channels, [out_channels1, out_channels2, out_channels3], kernel_sizes
            )
        
        # Build prior path
        if num_prior_features <= 0:
            self.prior_path = None
            self.prior_out_channels = 0
            self.prior_output_length = 0
        elif num_prior_features < MIN_FEATURES_FOR_CONV:
            self.prior_path = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_channels * num_prior_features, prior_channels1),
                nn.ReLU()
            )
            self.prior_out_channels = prior_channels1
            self.prior_output_length = 1
        else:
            self.prior_path, self.prior_out_channels, self.prior_output_length = self._build_conv_path(
                num_prior_features, in_channels, [prior_channels1, prior_channels2, prior_channels3], prior_kernel_sizes
            )
        
        self.total_output_length = self.main_output_length + self.prior_output_length
        
        if self.main_out_channels > 0 and self.prior_out_channels > 0:
            self.fusion = nn.Conv1d(self.main_out_channels + self.prior_out_channels, out_channels3, kernel_size=1)
            self.use_fusion = True
            self.final_channels = out_channels3
        else:
            self.fusion = None
            self.use_fusion = False
            self.final_channels = self.main_out_channels if self.main_out_channels > 0 else self.prior_out_channels

        self._to_linear = self._compute_fc_input_dim()
        
        fc_layers_list = []
        input_dim = self._to_linear
        for i in range(fc_layers):
            units = fc_units[i] if i < len(fc_units) else fc_units[-1]
            fc_layers_list.extend([
                nn.Linear(input_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = units
        fc_layers_list.append(nn.Linear(input_dim, out_dim))
        self.fc_layers = nn.Sequential(*fc_layers_list)
    
    def _build_conv_path(self, num_features, in_channels, channels_list, kernel_sizes):
        """Build convolutional path, returns (path, out_channels, output_length)"""
        if num_features < 16:
            num_layers = 2
        else:
            num_layers = 3
        
        layers = []
        in_ch = in_channels
        current_len = num_features
        
        for i in range(num_layers):
            out_ch = channels_list[min(i, len(channels_list) - 1)]
            
            k = min(kernel_sizes[min(i, len(kernel_sizes) - 1)], current_len)
            if k % 2 == 0 and k > 1:
                k -= 1
            k = max(1, k)
            
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            ])
            
            if current_len >= 4:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_len = current_len // 2
            
            in_ch = out_ch
        
        return nn.Sequential(*layers), in_ch, current_len
    
    def _compute_fc_input_dim(self):
        """Compute FC input dimension using eval mode"""
        self.eval()
        with torch.no_grad():
            x = torch.zeros(2, self.in_channels, self.seq_length)
            features = self._extract_features(x)
        self.train()
        return features.shape[1]
    
    def _extract_features(self, x):
        """Extract and combine features from both paths using padding"""
        batch_size = x.size(0)
        
        main_features = None
        prior_features = None
        
        # Main path
        if self.general_path is not None and self.num_main_features > 0:
            main_x = x[:, :, self.main_indices]
            main_features = self.general_path(main_x)
        
        # Prior path
        if self.prior_path is not None and self.num_prior_features > 0:
            prior_x = x[:, :, self.prior_indices]
            prior_features = self.prior_path(prior_x)
        
        if main_features is not None and main_features.dim() == 2:
            main_features = main_features.unsqueeze(-1)  # (batch, channels) -> (batch, channels, 1)
        if prior_features is not None and prior_features.dim() == 2:
            prior_features = prior_features.unsqueeze(-1)
        
        if main_features is not None and prior_features is not None:
            # Pad to total_output_length
            main_len = main_features.shape[2]
            prior_len = prior_features.shape[2]
            total_len = main_len + prior_len
            
            # Pad main at END
            main_pad = total_len - main_len
            if main_pad > 0:
                main_features = nn.functional.pad(main_features, (0, main_pad), mode='constant', value=0)
            
            # Pad prior at START
            prior_pad = total_len - prior_len
            if prior_pad > 0:
                prior_features = nn.functional.pad(prior_features, (prior_pad, 0), mode='constant', value=0)
            
            combined = torch.cat([main_features, prior_features], dim=1)
            
            if self.use_fusion:
                fused = self.fusion(combined)
            else:
                fused = combined
        elif main_features is not None:
            fused = main_features
        elif prior_features is not None:
            fused = prior_features
        else:
            fused = x.mean(dim=2, keepdim=True)
        
        return fused.view(batch_size, -1)
    
    def _get_prior_indices(self):
        if self.prior_features is None:
            return []
            
        indices = []
        missing_features = []
        
        if isinstance(self.prior_features, str):
            prior_ids = self.prior_features.split()
        elif isinstance(self.prior_features, (list, tuple)):
            prior_ids = self.prior_features
        else:
            return []

        import re
        name_to_idx = {}
        for idx, name in enumerate(self.feature_names):
            name_to_idx[name] = idx
        
        count_matched = 0
        for pid in prior_ids:
            try:
                if pid.isdigit():
                    idx = int(pid)
                    if 0 <= idx < self.seq_length:
                        indices.append(idx)
                        count_matched += 1
                    else:
                        pass
                else:
                    idx = name_to_idx.get(pid, -1)
                    if idx >= 0:
                        indices.append(idx)
                        count_matched += 1
                    else:
                        clean_pid = re.sub(r'[^0-9a-zA-Z_]', '_', pid)
                        idx = name_to_idx.get(clean_pid, -1)
                        if idx >= 0:
                            indices.append(idx)
                            count_matched += 1
                        else:
                            missing_features.append(pid)
            except ValueError:
                pass
        
        if missing_features and count_matched == 0 and len(prior_ids) > 0:
            pass
            
        return indices

    def forward(self, x):
        features = self._extract_features(x)
        return self.fc_layers(features)

def create_model(
    opts: ModelOpts = None,
    input_length: int = None,
    in_channels=None,
    out_channels1=None,
    out_channels2=None,
    out_channels3=None,
    prior_channels1=None,
    prior_channels2=None,
    prior_channels3=None,
    fc_layers=None,
    fc_units=None,
    out_dim=None,
    kernel_size=None,
    prior_kernel_size=None,
    dropout_prob=None,
    prior_features=None,
    feature_names=None
):
    if opts is None:
        opts = ModelOpts()
        
    if in_channels is not None:
        opts.in_channels = in_channels
    if out_channels1 is not None:
        opts.out_channels1 = out_channels1
    if out_channels2 is not None:
        opts.out_channels2 = out_channels2
    if out_channels3 is not None:
        opts.out_channels3 = out_channels3
    if prior_channels1 is not None:
        opts.prior_channels1 = prior_channels1
    if prior_channels2 is not None:
        opts.prior_channels2 = prior_channels2
    if prior_channels3 is not None:
        opts.prior_channels3 = prior_channels3
    if fc_layers is not None:
        opts.fc_layers = fc_layers
    if fc_units is not None:
        opts.fc_units = fc_units
    if out_dim is not None:
        opts.out_dim = out_dim
    if dropout_prob is not None:
        opts.dropout_prob = dropout_prob
    if prior_features is not None:
        opts.prior_features = prior_features

    if kernel_size is not None:
        if isinstance(kernel_size, (int, float)):
            opts.kernel_size = [kernel_size] * 3
        else:
            opts.kernel_size = kernel_size
            
    if prior_kernel_size is not None:
        if isinstance(prior_kernel_size, (int, float)):
            opts.prior_kernel_size = [prior_kernel_size] * 3
        else:
            opts.prior_kernel_size = prior_kernel_size
    
    kernel_sizes = opts.kernel_size
    prior_kernel_sizes = opts.prior_kernel_size
    
    if hasattr(opts, 'conv_kernel_size') and opts.conv_kernel_size:
        for i in range(min(len(opts.conv_kernel_size), 3)):
            kernel_sizes[i] = opts.conv_kernel_size[i]
            
    if hasattr(opts, 'prior_kernel_size') and opts.prior_kernel_size:
        for i in range(min(len(opts.prior_kernel_size), 3)):
            prior_kernel_sizes[i] = opts.prior_kernel_size[i]
        
    if input_length is None:
        raise ValueError("input_length must be specified")
    
    return PKDP(
        in_channels=opts.in_channels,
        out_channels1=opts.out_channels1,
        out_channels2=opts.out_channels2,
        out_channels3=opts.out_channels3,
        prior_channels1=opts.prior_channels1,
        prior_channels2=opts.prior_channels2,
        prior_channels3=opts.prior_channels3,
        fc_layers=opts.fc_layers,
        fc_units=opts.fc_units,
        out_dim=opts.out_dim,
        input_length=input_length,
        kernel_sizes=kernel_sizes,
        prior_kernel_sizes=prior_kernel_sizes,
        dropout_prob=opts.dropout_prob,
        prior_features=opts.prior_features,
        feature_names=feature_names
    )

def create_optimizer(model, learning_rate, optimizer_type='Adam'):
    if optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        log(ERROR, f"Unsupported optimizer type: {optimizer_type}")
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def create_loss_function():
    return nn.MSELoss()
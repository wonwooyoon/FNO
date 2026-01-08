#!/usr/bin/env python3
"""
Channel-wise Normalizer Core

순수한 normalization 로직만 담당:
- 채널별 transformation 적용
- 채널별 normalizer fit/transform
- Inverse transform
- Serialization

파일 I/O, 시각화 등은 포함하지 않음
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import sys
sys.path.append('./')
from neuraloperator.neuralop.data.transforms.normalizers import (
    UnitGaussianNormalizer,
    MinMaxNormalizer
)


class ChannelNormalizer:
    """
    간결하고 직관적인 채널별 normalizer

    Configuration format:
        input_config = [
            (idx, name, transform_type, normalizer_type),
            ...
        ]

        Examples:
            (0, 'Perm', 'log10', 'UnitGaussian')
            (1, 'Calcite', ('shifted_log', 1e-6), 'UnitGaussian')
            (5, 'Material', 'none', 'none')

    Output configuration:
        output_config = {
            'transform': 'log10' | 'none' | 'delta',
            'remove_t0': True,
            'mask_source': False  # Only for delta mode
        }
    """

    # ========================================================================
    # Static transformation functions
    # ========================================================================

    @staticmethod
    def transform_log10(x: torch.Tensor) -> torch.Tensor:
        """Apply log10 transformation"""
        return torch.log10(x)

    @staticmethod
    def transform_shifted_log(x: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Apply shifted log transformation: log10(x + eps) - log10(eps)

        This transformation:
        - Handles zeros and small values
        - Maps [0, inf) → [0, inf)
        - Preserves relative ordering
        """
        return torch.log10(x + eps) - np.log10(eps)

    @staticmethod
    def transform_delta(y: torch.Tensor, mask_source: bool = True, material_source_map: torch.Tensor = None) -> torch.Tensor:
        """
        Compute delta from t=0 initial state

        Args:
            y: (N, 1, nx, ny, nt) - raw output with t=0
            mask_source: Whether to mask source region
            material_source_map: (N, 1, nx, ny, nt) - material source map (1 where material==3)
                                 If provided, masks locations where map==1
                                 If None, uses legacy hard-coded indices [14:18, 14:18]

        Returns:
            delta: (N, 1, nx, ny, nt-1) - delta from t=0
        """
        initial = y[:, :, :, :, 0:1]  # (N, 1, nx, ny, 1) - reference at t=0
        delta = y - initial  # (N, 1, nx, ny, nt) - delta from t=0
        delta = delta[:, :, :, :, 1:]  # (N, 1, nx, ny, nt-1) - exclude t=0

        if mask_source:
            source_mask = material_source_map[:, :, :, :, 1:]  # (N, 1, nx, ny, nt-1)
            mask = (source_mask == 1)
            delta[mask] = 0

        return delta

    # ========================================================================
    # Initialization
    # ========================================================================

    def __init__(
        self,
        input_config: List[Tuple],
        output_mode: str,
        output_config: Dict
    ):
        """
        Initialize channel-wise normalizer

        Args:
            input_config: List of tuples with channel configuration
                Format 1 (simple): (idx, name, transform_type, normalizer_type)
                    transform_type: 'log10', 'none'
                    normalizer_type: 'UnitGaussian', 'MinMax', 'none'

                Format 2 (shifted_log): (idx, name, ('shifted_log', eps), normalizer_type)
                    eps: float epsilon value (e.g., 1e-6, 1e-9)

                Examples:
                    (0, 'Perm', 'log10', 'UnitGaussian')
                    (1, 'Calcite', ('shifted_log', 1e-6), 'UnitGaussian')
                    (5, 'Material', 'none', 'none')

            output_mode: 'log', 'raw', 'delta'
            output_config: Dict with 'transform', 'remove_t0', 'mask_source'
        """
        self.input_config = input_config
        self.output_mode = output_mode
        self.output_config = output_config

        # Will be populated during fit()
        self.input_normalizers = {}  # Dict[int, Normalizer]
        self.output_normalizer = None

    # ========================================================================
    # Input transformation
    # ========================================================================

    def apply_input_transforms(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply transformations to input channels

        Args:
            x_raw: (N, C, nx, ny, nt) - raw input data

        Returns:
            x_transformed: (N, C, nx, ny, nt) - transformed input
        """
        N, C, nx, ny, nt = x_raw.shape
        transformed_channels = []

        for config_item in self.input_config:
            idx, name, transform_type, _ = config_item[:4]
            ch = x_raw[:, idx:idx+1, :, :, :]  # (N, 1, nx, ny, nt)

            # Handle different transform types
            if transform_type == 'log10':
                ch = self.transform_log10(ch)

            elif transform_type == 'none':
                # Keep as-is
                pass

            elif isinstance(transform_type, tuple):
                # Format: ('shifted_log', eps)
                if transform_type[0] == 'shifted_log':
                    eps = transform_type[1]
                    ch = self.transform_shifted_log(ch, eps)
                else:
                    raise ValueError(f"Unknown tuple transform: {transform_type}")

            else:
                raise ValueError(f"Unknown transform_type: {transform_type}")

            transformed_channels.append(ch)

        return torch.cat(transformed_channels, dim=1)  # (N, C, nx, ny, nt)

    # ========================================================================
    # Output transformation
    # ========================================================================

    def apply_output_transform(self, y_raw: torch.Tensor, x_raw: torch.Tensor = None) -> torch.Tensor:
        """
        Apply transformation to output based on output_mode

        Args:
            y_raw: (N, 1, nx, ny, nt) - raw output with t=0
            x_raw: (N, C, nx, ny, nt) - raw input (optional, needed for material_source_map in delta mode)

        Returns:
            y_transformed: (N, 1, nx, ny, nt-1) - transformed output (t=0 removed)
        """
        N, C, nx, ny, nt = y_raw.shape
        assert C == 1, f"Expected 1 output channel, got {C}"

        transform_type = self.output_config['transform']

        if transform_type == 'log10':
            # Apply log10 transformation, remove t=0
            y = y_raw[:, :, :, :, 1:]  # Remove t=0
            y = torch.log10(y + 1e-12)  # Add small epsilon for stability

        elif transform_type == 'delta':
            # Compute delta from t=0, then remove t=0
            mask_source = self.output_config.get('mask_source', True)

            # Extract material_source_map from x_raw if available
            material_source_map = None
            if mask_source and x_raw is not None:
                # Channel 5 is Material_Source (one-hot encoding of material==3)
                material_source_map = x_raw[:, 5:6, :, :, :]  # (N, 1, nx, ny, nt)

            y = self.transform_delta(y_raw, mask_source=mask_source, material_source_map=material_source_map)

        elif transform_type == 'none':
            # Keep raw, but remove t=0
            y = y_raw[:, :, :, :, 1:]

        else:
            raise ValueError(f"Unknown output transform: {transform_type}")

        return y

    # ========================================================================
    # Fit normalizers
    # ========================================================================

    def fit(self, x_raw: torch.Tensor, y_raw: torch.Tensor, verbose: bool = True):
        """
        Fit normalizers on transformed data

        Args:
            x_raw: (N, C, nx, ny, nt) - raw input data
            y_raw: (N, 1, nx, ny, nt) - raw output data
            verbose: whether to print statistics
        """
        # 1. Apply transformations
        x_trans = self.apply_input_transforms(x_raw)
        y_trans = self.apply_output_transform(y_raw, x_raw=x_raw)

        if verbose:
            print(f"\n{'='*70}")
            print("Fitting Channel Normalizers")
            print(f"{'='*70}")
            print(f"Input shape:  {tuple(x_raw.shape)} → {tuple(x_trans.shape)}")
            print(f"Output shape: {tuple(y_raw.shape)} → {tuple(y_trans.shape)}")
            print(f"Output mode:  {self.output_mode}")
            print()

        # 2. Fit input normalizers
        for idx, name, _, norm_type in self.input_config:
            if norm_type == 'none':
                if verbose:
                    print(f"  [{idx:2d}] {name:20s} - No normalization")
                continue

            ch_data = x_trans[:, idx:idx+1, :, :, :]  # (N, 1, nx, ny, nt)

            # Create normalizer
            if norm_type == 'UnitGaussian':
                normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=0)
            elif norm_type == 'MinMax':
                normalizer = MinMaxNormalizer(dim=[0, 2, 3, 4], eps=0)
            else:
                raise ValueError(f"Unknown normalizer type: {norm_type}")

            # Fit normalizer
            normalizer.fit(ch_data)
            self.input_normalizers[idx] = normalizer

            # Print statistics
            if verbose:
                if norm_type == 'UnitGaussian':
                    mean_val = normalizer.mean.item() if normalizer.mean.numel() == 1 else normalizer.mean.mean().item()
                    std_val = normalizer.std.item() if normalizer.std.numel() == 1 else normalizer.std.mean().item()
                    print(f"  [{idx:2d}] {name:20s} - UnitGaussian: mean={mean_val:10.4e}, std={std_val:10.4e}")
                elif norm_type == 'MinMax':
                    min_val = normalizer.data_min.item() if normalizer.data_min.numel() == 1 else normalizer.data_min.min().item()
                    max_val = normalizer.data_max.item() if normalizer.data_max.numel() == 1 else normalizer.data_max.max().item()
                    print(f"  [{idx:2d}] {name:20s} - MinMax:       min={min_val:10.4e}, max={max_val:10.4e}")

        # 3. Fit output normalizer
        self.output_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=0)
        self.output_normalizer.fit(y_trans)

        if verbose:
            print()
            mean_val = self.output_normalizer.mean.item() if self.output_normalizer.mean.numel() == 1 else self.output_normalizer.mean.mean().item()
            std_val = self.output_normalizer.std.item() if self.output_normalizer.std.numel() == 1 else self.output_normalizer.std.mean().item()
            print(f"  [ 0] {'Uranium':20s} - UnitGaussian: mean={mean_val:10.4e}, std={std_val:10.4e}")
            print(f"{'='*70}\n")

    # ========================================================================
    # Transform (apply normalization)
    # ========================================================================

    def transform(
        self,
        x_raw: torch.Tensor,
        y_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations and normalization

        Args:
            x_raw: (N, C, nx, ny, nt) - raw input
            y_raw: (N, 1, nx, ny, nt) - raw output

        Returns:
            x_norm: (N, C, nx, ny, nt-1) - normalized input (t=0 removed)
            y_norm: (N, 1, nx, ny, nt-1) - normalized output (t=0 removed)
        """
        # 1. Apply transformations
        x_trans = self.apply_input_transforms(x_raw)
        y_trans = self.apply_output_transform(y_raw, x_raw=x_raw)

        # 2. Remove t=0 from input to match output timesteps
        x_trans = x_trans[:, :, :, :, 1:]  # (N, C, nx, ny, nt-1)

        # 3. Apply normalization to each input channel
        x_norm_channels = []
        for idx, _, _, norm_type in self.input_config:
            ch = x_trans[:, idx:idx+1, :, :, :]

            if idx in self.input_normalizers:
                # Apply normalization
                ch = self.input_normalizers[idx].transform(ch)
            # else: no normalization (one-hot encoded channels)

            x_norm_channels.append(ch)

        x_norm = torch.cat(x_norm_channels, dim=1)

        # 4. Normalize output
        y_norm = self.output_normalizer.transform(y_trans)

        return x_norm, y_norm

    # ========================================================================
    # Inverse transform
    # ========================================================================

    def inverse_transform_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Complete inverse transform: normalized → transformed → raw

        Args:
            y_norm: (N, 1, nx, ny, nt) - normalized output

        Returns:
            y_raw: (N, 1, nx, ny, nt) - raw physical values
        """
        # Get device from input
        device = y_norm.device

        # Ensure output_normalizer is on same device
        if hasattr(self.output_normalizer, 'mean'):
            if self.output_normalizer.mean.device != device:
                self.output_normalizer.mean = self.output_normalizer.mean.to(device)
                self.output_normalizer.std = self.output_normalizer.std.to(device)

        # 1. Denormalize
        y_trans = self.output_normalizer.inverse_transform(y_norm)

        # 2. Inverse transformation
        transform_type = self.output_config['transform']

        if transform_type == 'log10':
            # Inverse log: log10(C) → C
            y_raw = torch.pow(10, y_trans)

        elif transform_type == 'delta':
            # Delta mode: cannot fully reverse without t=0 reference
            # Return in transformed space (delta values)
            y_raw = y_trans

        elif transform_type == 'none':
            # No transformation, already in raw space
            y_raw = y_trans

        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        return y_raw

    # ========================================================================
    # Device management
    # ========================================================================

    def to(self, device):
        """Move all normalizers to device"""
        for normalizer in self.input_normalizers.values():
            normalizer.to(device)
        if self.output_normalizer is not None:
            self.output_normalizer.to(device)
        return self

    def cpu(self):
        """Move all normalizers to CPU"""
        return self.to('cpu')

    def cuda(self):
        """Move all normalizers to CUDA"""
        return self.to('cuda')


# ============================================================================
# Outlet Normalizer
# ============================================================================

class OutletNormalizer:
    """
    Outlet 데이터 전용 normalizer

    Transformation pipeline:
    1. Take absolute value (outlet is negative in PFLOTRAN)
    2. Apply log10(x + eps)
    3. Gaussian normalization (zero mean, unit variance)

    Usage:
        normalizer = OutletNormalizer()
        normalizer.fit(y_outlet_raw)
        y_outlet_norm = normalizer.transform(y_outlet_raw)
        y_outlet_recovered = normalizer.inverse_transform(y_outlet_norm)
    """

    def __init__(self, device='cpu'):
        """
        Initialize OutletNormalizer

        Args:
            device: Device to run on ('cpu' or 'cuda')
        """
        self.mean = None
        self.std = None
        self.eps = 1e-20
        self.device = device

    def fit(self, y_outlet: torch.Tensor, verbose: bool = True) -> 'OutletNormalizer':
        """
        Fit normalizer on outlet data

        Args:
            y_outlet: (N, nt) raw outlet data (negative values from PFLOTRAN)
            verbose: Print statistics

        Returns:
            self
        """
        if y_outlet.dim() != 2:
            raise ValueError(f"Expected 2D tensor (N, nt), got shape {y_outlet.shape}")

        # Transform: absolute → log10
        y_abs = torch.abs(y_outlet)
        y_log = torch.log10(y_abs + self.eps)

        # Compute statistics
        self.mean = y_log.mean().item()
        self.std = y_log.std().item()

        if verbose:
            print(f"  Outlet normalizer fitted:")
            print(f"    Original range: [{y_outlet.min():.2e}, {y_outlet.max():.2e}]")
            print(f"    After abs+log:  [{y_log.min():.4f}, {y_log.max():.4f}]")
            print(f"    Mean: {self.mean:.4f}, Std: {self.std:.4f}")

        return self

    def transform(self, y_outlet: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation and normalization

        Args:
            y_outlet: (N, nt) raw outlet data

        Returns:
            y_outlet_norm: (N, nt) normalized outlet data
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        y_abs = torch.abs(y_outlet)
        y_log = torch.log10(y_abs + self.eps)
        y_norm = (y_log - self.mean) / self.std

        return y_norm

    def inverse_transform(self, y_outlet_norm: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization and transformation

        Args:
            y_outlet_norm: (N, nt) normalized outlet data

        Returns:
            y_outlet: (N, nt) raw outlet data (negative values)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        y_log = y_outlet_norm * self.std + self.mean
        y_abs = 10 ** y_log - self.eps
        y_outlet = -y_abs  # Restore negative sign

        return y_outlet

    def cpu(self):
        """Move to CPU (for consistency with ChannelNormalizer)"""
        self.device = 'cpu'
        return self

    def to(self, device):
        """Move to device (for consistency with ChannelNormalizer)"""
        self.device = device
        return self

    def __repr__(self):
        """String representation"""
        if self.mean is None:
            return "OutletNormalizer(not fitted)"
        return f"OutletNormalizer(mean={self.mean:.4f}, std={self.std:.4f}, eps={self.eps})"

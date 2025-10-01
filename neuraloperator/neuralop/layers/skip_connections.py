import torch
from torch import nn


def skip_connection(
    in_features, out_features, n_dim=2, bias=False, skip_type="soft-gating"
):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, optional
        whether to use a bias, by default False
    skip_type : {'identity', 'linear', 'soft-gating', 'attention'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return Flattened1dConv(in_channels=in_features,
                               out_channels=out_features,
                               kernel_size=1,
                               bias=bias,)
    elif skip_type.lower() == "identity":
        return nn.Identity()
    elif skip_type.lower() == "attention":
        return SelfAttentionSkip(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'identity', 'attention'}."
        )


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}, "
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x

class Flattened1dConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):
        """Flattened3dConv is a Conv-based skip layer for
        input tensors of ndim > 3 (batch, channels, d1, ...) that flattens all dimensions 
        past the batch and channel dims into one dimension, applies the Conv,
        and un-flattens.

        Parameters
        ----------
        in_channels : int
            in_channels of Conv1d
        out_channels : int
            out_channels of Conv1d
        kernel_size : int
            kernel_size of Conv1d
        bias : bool, optional
            bias of Conv3d, by default False
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              bias=bias)
    def forward(self, x):
        # x.shape: b, c, x1, ..., xn x_ndim > 1
        size = list(x.shape)
        # flatten everything past 1st data dim
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        # reshape x into an Nd tensor b, c, x1, x2, ...
        x = x.view(size[0], self.conv.out_channels, *size[2:])
        return x


class SelfAttentionSkip(nn.Module):
    """Galerkin attention based skip connection

    Applies Galerkin self-attention mechanism where K and V are layer normalized
    before computing their interaction, followed by interaction with Q.
    This follows the Galerkin attention approach: Q(K^T V)/n instead of (QK^T)V/n.

    Parameters
    ----------
    in_features : int
        number of input features (channels)
    out_features : int, optional
        number of output features, by default None (same as in_features)
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
    bias : bool, default is False
        whether to use bias in linear projections
    dropout : float, default is 0.1
        dropout probability
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.n_dim = n_dim

        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_features, self.out_features, bias=bias)
        self.k_proj = nn.Linear(in_features, in_features, bias=bias)
        self.v_proj = nn.Linear(in_features, in_features, bias=bias)

        # Layer normalization for K and V (pre-dot-product)
        self.k_norm = nn.LayerNorm(in_features)
        self.v_norm = nn.LayerNorm(in_features)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply Galerkin attention on spatial dimensions

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, dim1, dim2, ..., dimN)
            where N = n_dim

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_features, dim1, dim2, ..., dimN)
        """
        original_shape = x.shape
        batch_size = original_shape[0]
        channels = original_shape[1]
        spatial_dims = original_shape[2:]

        # Flatten spatial dimensions: (B, C, D1*D2*...*DN)
        total_spatial = 1
        for dim in spatial_dims:
            total_spatial *= dim

        # Reshape to (B, spatial_sequence, channels)
        x_reshaped = x.view(batch_size, channels, total_spatial)
        x_reshaped = x_reshaped.transpose(1, 2)  # (B, spatial_sequence, channels)

        # Compute Q, K, V projections
        Q = self.q_proj(x_reshaped)  # (B, spatial_sequence, out_features)
        K = self.k_proj(x_reshaped)  # (B, spatial_sequence, in_features)
        V = self.v_proj(x_reshaped)  # (B, spatial_sequence, in_features)

        # Apply layer normalization to K and V
        K = self.k_norm(K)
        V = self.v_norm(V)

        # Galerkin attention: Q(K^T V) / n
        # K^T: (B, in_features, spatial_sequence)
        # V: (B, spatial_sequence, in_features)
        # K^T V: (B, in_features, in_features)
        KT_V = torch.bmm(K.transpose(1, 2), V)  # (B, in_features, in_features)

        # Q: (B, spatial_sequence, out_features)
        # Q(K^T V): (B, spatial_sequence, in_features)
        attn_output = torch.bmm(Q, KT_V.transpose(1, 2))  # (B, spatial_sequence, in_features)

        # Scale by sequence length
        attn_output = attn_output / total_spatial

        # Apply dropout
        attn_output = self.dropout(attn_output)

        # If out_features != in_features, we need to project to correct size
        if self.out_features != self.in_features:
            # Take only the first out_features dimensions
            attn_output = attn_output[:, :, :self.out_features]

        # Reshape back to original spatial dimensions
        attn_output = attn_output.transpose(1, 2)  # (B, out_features, spatial_sequence)
        output_shape = (batch_size, self.out_features) + spatial_dims
        attn_output = attn_output.view(output_shape)

        return attn_output


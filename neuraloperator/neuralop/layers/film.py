import torch
import torch.nn as nn
from typing import Optional, Tuple


class FiLMLayer(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) layer for FNO feature maps.

    Given a latent feature map `x` and metadata vector `meta`,
    this layer predicts per-channel (gamma, beta) and applies:
        y = x * Gamma + Beta

    where Gamma, Beta are broadcasted to spatial dimensions.

    Args
    ----
    meta_dim : int
        Dimension of the metadata vector (B, meta_dim).
    feature_channels : int
        Number of channels in `x` to be modulated (C).
    n_dim : int
        Number of spatial dimensions of `x` (1, 2, or 3).
    width : int, default=256
        Hidden units of the MLP that encodes metadata.
    depth : int, default=1
        Number of hidden layers (Linear(+Norm)+Act(+Dropout)) in the MLP.
    activation : str, default="gelu"
        One of {"gelu", "relu", "silu", "tanh"}.
    use_layernorm : bool, default=False
        If True, applies LayerNorm after each hidden Linear.
    dropout : float, default=0.0
        Dropout probability applied after activation (per hidden layer).
    zero_init_head : bool, default=True
        If True, initializes the final Linear to zeros → starts near identity
        (gamma≈1, beta≈0).
    gamma_clamp : Optional[float], default=0.1
        If not None, uses gamma = 1 + gamma_clamp * tanh(gamma_raw).
        If None, uses gamma = 1 + gamma_raw.

    Shapes
    ------
    x    : (B, C, *spatial)   # *spatial has length n_dim
    meta : (B, meta_dim)
    out  : (B, C, *spatial)
    """

    def __init__(self,
                 meta_dim: int,
                 feature_channels: int,
                 n_dim: int,
                 width: int = 256,
                 depth: int = 1,
                 activation: str = "gelu",
                 use_layernorm: bool = False,
                 dropout: float = 0.0,
                 zero_init_head: bool = True,
                 gamma_clamp: Optional[float] = 0.1):
        super().__init__()
        assert n_dim in (1, 2, 3), f"n_dim must be 1, 2 or 3, got {n_dim}"
        self.n_dim = n_dim
        self.C = feature_channels
        self.gamma_clamp = gamma_clamp

        act_map = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation '{activation}'. "
                             f"Choose from {list(act_map.keys())}.")
        Act = act_map[activation]

        # --- Meta encoder (MLP) ---
        layers = []
        in_dim = meta_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            if use_layernorm:
                layers.append(nn.LayerNorm(width))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Head produces [gamma_raw | beta] of size 2*C
        self.head = nn.Linear(in_dim, 2 * feature_channels)

        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _broadcast_to_spatial(bc: torch.Tensor,
                              spatial_shape: Tuple[int, ...],
                              n_dim: int) -> torch.Tensor:
        """
        Turn (B, C) into (B, C, *spatial) by unsqueezing n_dim times and expanding.
        """
        for _ in range(n_dim):
            bc = bc.unsqueeze(-1)  # (B, C, 1, 1[, 1])
        return bc.expand(-1, -1, *spatial_shape)

    def forward(self,
                x: torch.Tensor,
                meta: torch.Tensor,
                return_params: bool = False):
        """
        Apply FiLM modulation.

        Parameters
        ----------
        x : torch.Tensor
            Latent feature map, shape (B, C, *spatial).
        meta : torch.Tensor
            Metadata, shape (B, meta_dim).
        return_params : bool
            If True, also returns per-sample (gamma, beta) as (B, C).

        Returns
        -------
        y : torch.Tensor
            Modulated feature map, same shape as x.
        (optional) gamma, beta : torch.Tensor, torch.Tensor
            Per-sample parameters (B, C) before spatial broadcasting.
        """
        B, C = x.shape[:2]
        assert C == self.C, f"feature_channels mismatch: expected {self.C}, got {C}"
        assert meta.dim() == 2 and meta.shape[0] == B, \
            f"meta must be (B, meta_dim), got {tuple(meta.shape)}"

        gb = self.head(self.backbone(meta))           # (B, 2C)
        gamma_raw, beta = torch.chunk(gb, 2, dim=1)   # (B, C), (B, C)

        if self.gamma_clamp is None:
            gamma = 1.0 + gamma_raw
        else:
            gamma = 1.0 + self.gamma_clamp * torch.tanh(gamma_raw)

        Gamma = self._broadcast_to_spatial(gamma, x.shape[2:], self.n_dim)
        Beta  = self._broadcast_to_spatial(beta,  x.shape[2:], self.n_dim)

        y = x * Gamma + Beta

        if return_params:
            return y, gamma, beta
        return y
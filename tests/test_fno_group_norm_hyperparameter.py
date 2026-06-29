import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

import FNO
import FNO_outlet


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "x": torch.zeros(11, 2, 2, 2),
            "y": torch.zeros(1, 2, 2, 2),
        }


class FakeTFNO(nn.Module):
    captured_kwargs = None

    def __init__(self, **kwargs):
        super().__init__()
        FakeTFNO.captured_kwargs = kwargs
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4], device=x.device)


class TinyOutletDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "x": torch.zeros(11, 2, 2, 2),
            "y": torch.zeros(2),
        }


class FakeOutletTFNO(nn.Module):
    captured_kwargs = None

    def __init__(self, **kwargs):
        super().__init__()
        FakeOutletTFNO.captured_kwargs = kwargs
        self.positional_embedding = nn.Identity()
        self.lifting = nn.Identity()
        self.fno_blocks = nn.Identity()
        self.domain_padding = None
        self.hidden_channels = kwargs["hidden_channels"]
        self.n_layers = kwargs["n_layers"]


def test_fno_optuna_search_space_exposes_group_norm_option():
    assert FNO.CONFIG["OPTUNA_SEARCH_SPACE"]["norm_options"] == [None, "group_norm"]


def test_fno_outlet_optuna_search_space_exposes_group_norm_option():
    assert FNO_outlet.CONFIG["OPTUNA_SEARCH_SPACE"]["norm_options"] == [None, "group_norm"]


def test_create_model_passes_group_norm_to_tfno(monkeypatch):
    monkeypatch.setattr(FNO, "TFNO", FakeTFNO)

    FNO.create_model(
        config=FNO.CONFIG,
        train_dataset=TinyDataset(),
        val_dataset=TinyDataset(),
        test_dataset=TinyDataset(),
        device="cpu",
        n_modes=(2, 2, 2),
        hidden_channels=4,
        n_layers=1,
        domain_padding=(0.1, 0.1, 0.1),
        train_batch_size=1,
        l2_weight=0.0,
        channel_mlp_expansion=1.0,
        channel_mlp_skip="linear",
        norm="group_norm",
    )

    assert FakeTFNO.captured_kwargs["norm"] == "group_norm"


def test_create_outlet_model_passes_group_norm_to_base_tfno(monkeypatch):
    monkeypatch.setattr(FNO_outlet, "TFNO", FakeOutletTFNO)

    FNO_outlet.create_model(
        config=FNO_outlet.CONFIG,
        train_dataset=TinyOutletDataset(),
        val_dataset=TinyOutletDataset(),
        test_dataset=TinyOutletDataset(),
        device="cpu",
        n_modes=(2, 2, 2),
        hidden_channels=4,
        n_layers=1,
        domain_padding=(0.1, 0.1, 0.1),
        train_batch_size=1,
        l2_weight=0.0,
        channel_mlp_expansion=1.0,
        channel_mlp_skip="linear",
        projection_mlp_hidden=4,
        projection_mlp_layers=2,
        projection_mlp_activation="gelu",
        projection_mlp_dropout=0.0,
        norm="group_norm",
    )

    assert FakeOutletTFNO.captured_kwargs["norm"] == "group_norm"


def test_create_outlet_model_from_legacy_params_defaults_norm_to_none(monkeypatch):
    monkeypatch.setattr(FNO_outlet, "TFNO", FakeOutletTFNO)
    params = {
        "n_modes_1": 2,
        "n_modes_2": 2,
        "n_modes_3": 2,
        "hidden_channels": 4,
        "n_layers": 1,
        "domain_padding": (0.1, 0.1, 0.1),
        "train_batch_size": 1,
        "l2_weight": 0.0,
        "channel_mlp_expansion": 1.0,
        "channel_mlp_skip": "linear",
        "projection_mlp_hidden": 4,
        "projection_mlp_layers": 2,
        "projection_mlp_activation": "gelu",
        "projection_mlp_dropout": 0.0,
    }

    FNO_outlet.create_model_from_params(
        FNO_outlet.CONFIG,
        TinyOutletDataset(),
        TinyOutletDataset(),
        TinyOutletDataset(),
        "cpu",
        params,
    )

    assert FakeOutletTFNO.captured_kwargs["norm"] is None

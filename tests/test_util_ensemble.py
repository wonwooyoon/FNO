import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_ensemble import (
    EnsemblePredictor,
    fixed_test_split,
    make_member_split,
    resolve_domain_padding,
    save_ensemble_manifest,
)


class ConstantModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value, dtype=x.dtype, device=x.device)


def test_ensemble_predictor_returns_mean_prediction():
    predictor = EnsemblePredictor(
        models=[ConstantModel(1.0), ConstantModel(3.0), ConstantModel(5.0)],
        device="cpu",
    )

    pred = predictor(torch.zeros(4, 2))

    assert pred.shape == (4, 1)
    assert torch.allclose(pred, torch.full((4, 1), 3.0))


def test_fixed_test_split_keeps_test_indices_out_of_member_splits():
    dataset = TensorDataset(torch.arange(20).float().unsqueeze(1))
    trainval, test, split = fixed_test_split(dataset, test_size=0.25, random_state=7)

    member_a_train, member_a_val, member_a_split = make_member_split(
        trainval,
        trainval_original_indices=split.trainval_indices,
        val_size_relative=0.2,
        random_state=100,
    )
    member_b_train, member_b_val, member_b_split = make_member_split(
        trainval,
        trainval_original_indices=split.trainval_indices,
        val_size_relative=0.2,
        random_state=101,
    )

    test_indices = set(split.test_indices.tolist())
    assert len(trainval) == 15
    assert len(test) == 5
    assert test_indices.isdisjoint(member_a_split.train_indices)
    assert test_indices.isdisjoint(member_a_split.val_indices)
    assert test_indices.isdisjoint(member_b_split.train_indices)
    assert test_indices.isdisjoint(member_b_split.val_indices)
    assert set(member_a_split.train_indices.tolist()) != set(member_b_split.train_indices.tolist())
    assert len(member_a_train) + len(member_a_val) == len(trainval)


def test_resolve_domain_padding_converts_optuna_index_to_tuple():
    search_space = {
        "domain_padding_options": [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1)],
    }
    params = {
        "domain_padding_idx": 1,
        "n_modes_1": 4,
    }

    resolved = resolve_domain_padding(params, search_space)

    assert resolved["domain_padding"] == (0.2, 0.1, 0.1)
    assert "domain_padding_idx" not in resolved
    assert resolved["n_modes_1"] == 4


def test_save_ensemble_manifest_records_member_paths_and_params(tmp_path):
    manifest_path = tmp_path / "ensemble_manifest.json"
    save_ensemble_manifest(
        manifest_path=manifest_path,
        model_kind="fno",
        params={"train_batch_size": 2},
        member_records=[
            {
                "member_id": 0,
                "seed": 42,
                "model_state_path": "member_000/best_model_state_dict.pt",
                "loss_history_path": "member_000/loss_history.pt",
            }
        ],
        test_indices=[1, 2],
        config_subset={"N_EPOCHS": 1},
    )

    data = json.loads(manifest_path.read_text())
    assert data["format_version"] == 1
    assert data["model_kind"] == "fno"
    assert data["params"]["train_batch_size"] == 2
    assert data["members"][0]["member_id"] == 0
    assert data["test_indices"] == [1, 2]

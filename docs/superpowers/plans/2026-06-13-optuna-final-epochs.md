# Optuna Final Epoch Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure Optuna trial epochs independently from final single/ensemble training epochs.

**Architecture:** Add one shared helper in `util_training.py` that returns a shallow config copy with `N_EPOCHS` replaced by `TRAINING_CONFIG["optuna_n_epochs"]` when present. Use that helper only inside Optuna objective training in `FNO.py` and `FNO_outlet.py`; final ensemble training keeps the original config.

**Tech Stack:** Python, PyTorch training utilities, pytest.

---

### Task 1: Add Optuna Epoch Config Tests

**Files:**
- Create: `tests/test_optuna_epoch_config.py`
- Modify: none

- [ ] **Step 1: Write failing tests**

```python
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FNO_SRC = REPO_ROOT / "src" / "FNO"
if str(FNO_SRC) not in sys.path:
    sys.path.insert(0, str(FNO_SRC))

from util_training import config_for_optuna_epochs


def test_config_for_optuna_epochs_uses_training_config_override_without_mutating_original():
    config = {
        "N_EPOCHS": 100,
        "TRAINING_CONFIG": {
            "optuna_n_epochs": 12,
        },
    }

    optuna_config = config_for_optuna_epochs(config)

    assert optuna_config["N_EPOCHS"] == 12
    assert config["N_EPOCHS"] == 100
    assert optuna_config["TRAINING_CONFIG"] is config["TRAINING_CONFIG"]


def test_config_for_optuna_epochs_falls_back_to_final_epochs_when_missing():
    config = {
        "N_EPOCHS": 80,
        "TRAINING_CONFIG": {},
    }

    optuna_config = config_for_optuna_epochs(config)

    assert optuna_config["N_EPOCHS"] == 80
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_optuna_epoch_config.py -q`
Expected: FAIL because `config_for_optuna_epochs` is not defined.

### Task 2: Add Shared Helper

**Files:**
- Modify: `src/FNO/util_training.py`
- Test: `tests/test_optuna_epoch_config.py`

- [ ] **Step 1: Implement helper**

```python
def config_for_optuna_epochs(config: Dict) -> Dict:
    """Return a config copy with Optuna-specific N_EPOCHS applied."""
    optuna_n_epochs = config.get("TRAINING_CONFIG", {}).get("optuna_n_epochs", config["N_EPOCHS"])
    return {**config, "N_EPOCHS": int(optuna_n_epochs)}
```

- [ ] **Step 2: Run tests to verify pass**

Run: `pytest tests/test_optuna_epoch_config.py -q`
Expected: PASS.

### Task 3: Wire FNO and Outlet Optuna Training

**Files:**
- Modify: `src/FNO/FNO.py`
- Modify: `src/FNO/FNO_outlet.py`
- Test: `tests/test_fno_group_norm_hyperparameter.py`, `tests/test_util_ensemble.py`, `tests/test_optuna_epoch_config.py`

- [ ] **Step 1: Import helper**

Update both model files to import `config_for_optuna_epochs` from `util_training`.

- [ ] **Step 2: Add default config values**

Add `optuna_n_epochs` to each `TRAINING_CONFIG` dictionary.

- [ ] **Step 3: Use helper in Optuna objective**

In each Optuna objective, create `trial_config = config_for_optuna_epochs(config)` immediately before `train_model_to_dir()` and pass `trial_config` instead of `config`.

- [ ] **Step 4: Run focused verification**

Run: `pytest tests/test_optuna_epoch_config.py tests/test_fno_group_norm_hyperparameter.py tests/test_util_ensemble.py -q`
Expected: PASS.

- [ ] **Step 5: Compile modified modules**

Run: `python3 -m py_compile src/FNO/util_training.py src/FNO/FNO.py src/FNO/FNO_outlet.py`
Expected: exit code 0.

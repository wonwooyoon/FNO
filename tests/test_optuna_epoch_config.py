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

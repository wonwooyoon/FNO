# Optuna and Final Epoch Separation Design

## Goal
Allow Optuna trial training and final single/ensemble training to use different epoch counts configured from the existing Python config dictionaries.

## Current Behavior
`src/FNO/FNO.py` and `src/FNO/FNO_outlet.py` pass the same `CONFIG` dictionary to both Optuna trial training and final ensemble training. The shared training loop in `src/FNO/util_training.py` reads `config["N_EPOCHS"]`, so Optuna and final analysis currently train for the same maximum number of epochs.

## Design
Add `TRAINING_CONFIG["optuna_n_epochs"]` to both `FNO.py` and `FNO_outlet.py`. During Optuna objective execution, create a shallow config copy with top-level `N_EPOCHS` replaced by `optuna_n_epochs`, and pass that copy only to `train_model_to_dir()`.

Final single/ensemble training will continue to pass the original `CONFIG`, so it keeps using top-level `N_EPOCHS`. If `optuna_n_epochs` is absent, Optuna falls back to `N_EPOCHS` to preserve compatibility with older configs.

## Files
- `src/FNO/util_training.py`: add a small helper that resolves and applies an Optuna-only epoch override without mutating the original config.
- `src/FNO/FNO.py`: add default config value and use the helper in Optuna trial training.
- `src/FNO/FNO_outlet.py`: add default config value and use the helper in Optuna trial training.
- `tests/test_optuna_epoch_config.py`: verify fallback behavior, override behavior, and no mutation of the original config.

## Testing
Unit tests will cover the helper directly. A focused compile check will verify that the modified FNO modules remain syntactically valid.

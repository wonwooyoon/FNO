"""
Common Training Utilities for FNO Models

This module provides shared training logic used by both FNO.py and FNO_outlet.py:
- Generic training loop with early stopping
- Generic model evaluation
- Loss tracking and visualization

Refactored from FNO.py and FNO_outlet.py to eliminate code duplication.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ==============================================================================
# Training Functions
# ==============================================================================

def train_model_generic(
    config: Dict,
    device: str,
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    output_dir: Union[str, Path],
    verbose: bool = True
) -> nn.Module:
    """
    Generic training loop for FNO models with early stopping and loss tracking.

    This function works for both spatial field prediction (FNO.py) and
    vector output prediction (FNO_outlet.py).

    Args:
        config: Configuration dictionary
        device: Device to use (cuda/cpu)
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        output_dir: Directory to save model and loss history
        verbose: Whether to print training progress

    Returns:
        Trained model (with best weights loaded)
    """

    if verbose:
        print(f"\nStarting model training for {config['N_EPOCHS']} epochs...")

    # Training setup
    best_val_loss = float('inf')
    patience = 0
    early_stopping_patience = config['SCHEDULER_CONFIG']['early_stopping']

    # Track losses for each epoch
    train_losses = []
    val_losses = []

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['N_EPOCHS']):
        # Training phase
        model.train()
        total_train_loss = 0
        train_count = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_count += 1

        train_loss = total_train_loss / train_count
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)

                pred = model(x)
                loss = loss_fn(pred, y)
                total_val_loss += loss.item()
                val_count += 1

        val_loss = total_val_loss / val_count
        val_losses.append(val_loss)

        # Print losses for every epoch
        if verbose:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model_state_dict.pt')
            patience = 0
            if verbose:
                print(f"    New best model saved! Val loss: {val_loss:.6f}")
        else:
            patience += 1

        # Early stopping
        if patience >= early_stopping_patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs")
            break

        # Update learning rate
        scheduler.step()

    # Save loss history
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': list(range(len(train_losses)))
    }
    torch.save(loss_history, output_dir / 'loss_history.pt')

    # Plot training curves
    plt.figure(figsize=(10, 6))
    epochs_range = range(len(train_losses))
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    loss_plot_path = output_dir / 'loss_curves.png'
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nTraining completed!")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Loss history saved to: {output_dir / 'loss_history.pt'}")
        print(f"Loss curves saved to: {loss_plot_path}")

    # Load and return best model
    model.load_state_dict(torch.load(output_dir / 'best_model_state_dict.pt',
                                     map_location=device, weights_only=False))
    return model


def model_evaluation_generic(
    config: Dict,
    device: str,
    model: nn.Module,
    test_loader,
    loss_fn,
    output_dir: Union[str, Path],
    compute_mse: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Generic model evaluation on test set.

    Works for both spatial and vector outputs.

    Args:
        config: Configuration dictionary
        device: Device to use (cuda/cpu)
        model: Trained model to evaluate
        test_loader: Test data loader
        loss_fn: Loss function
        output_dir: Directory to save evaluation results
        compute_mse: Whether to additionally compute MSE (when using LpLoss)
        verbose: Whether to print evaluation results

    Returns:
        Dictionary containing evaluation results
    """

    if verbose:
        print(f"\nEvaluating model on test set...")

    # Test evaluation
    model.eval()
    total_test_loss = 0
    total_test_mse_loss = 0
    test_count = 0

    # Create MSE loss function for additional metric when requested
    mse_loss_fn = torch.nn.MSELoss() if compute_mse else None

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_test_loss += loss.item()

            # Calculate MSE loss additionally if requested
            if mse_loss_fn is not None:
                mse_loss = mse_loss_fn(pred, y)
                total_test_mse_loss += mse_loss.item()

            test_count += 1

    final_test_loss = total_test_loss / test_count
    final_test_mse_loss = total_test_mse_loss / test_count if mse_loss_fn is not None else None

    # Create evaluation results
    eval_results = {
        'test_loss': final_test_loss,
        'test_mse_loss': final_test_mse_loss if final_test_mse_loss is not None else None
    }

    # Save evaluation results
    output_dir = Path(output_dir)
    eval_results_path = output_dir / 'evaluation_results.pt'
    torch.save(eval_results, eval_results_path)

    if verbose:
        print(f"Model Evaluation Results:")
        loss_type = config.get('LOSS_CONFIG', {}).get('loss_type', 'unknown')

        if loss_type == 'l2' and final_test_mse_loss is not None:
            l2_p = config.get('LOSS_CONFIG', {}).get('l2_p', 2)
            print(f"  Test Loss (L{l2_p}): {final_test_loss:.6f}")
            print(f"  Test Loss (MSE): {final_test_mse_loss:.6f}")
        elif loss_type == 'mse':
            print(f"  Test Loss (MSE): {final_test_loss:.6f}")
        else:
            print(f"  Test Loss: {final_test_loss:.6f}")

        print(f"Evaluation results saved to: {eval_results_path}")

    return eval_results

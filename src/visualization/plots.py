"""Plotting utilities for training history and evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR


def plot_training_curves(
    bc_history: pd.DataFrame,
    dpo_history: pd.DataFrame,
    save_dir: Path | str = FIGURES_DIR,
) -> None:
    """Plot training/validation loss and accuracy for BC and DPO.

    Parameters
    ----------
    bc_history : pd.DataFrame
        Columns: ``loss``, ``val_loss``, ``accuracy``, ``val_accuracy``.
    dpo_history : pd.DataFrame
        Same schema as *bc_history*.
    save_dir : Path
        Directory where figures are saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Loss curves ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(bc_history["loss"], label="BC Train Loss")
    axes[0].plot(bc_history["val_loss"], label="BC Val Loss")
    axes[0].set_title("Behavior Cloning — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dpo_history["loss"], label="DPO Train Loss")
    axes[1].plot(dpo_history["val_loss"], label="DPO Val Loss")
    axes[1].set_title("Direct Preference Optimization — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Accuracy curves ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(bc_history["accuracy"], label="BC Train Acc")
    axes[0].plot(bc_history["val_accuracy"], label="BC Val Acc")
    axes[0].set_title("Behavior Cloning — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dpo_history["accuracy"], label="DPO Train Acc")
    axes[1].plot(dpo_history["val_accuracy"], label="DPO Val Acc")
    axes[1].set_title("Direct Preference Optimization — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "training_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_dir}")


def plot_f1_comparison(
    bc_f1: float,
    dpo_f1: float,
    save_dir: Path | str = FIGURES_DIR,
) -> None:
    """Bar chart comparing F1 scores of BC vs DPO on the golden dataset.

    Parameters
    ----------
    bc_f1 : float
        Behavior Cloning F1 score.
    dpo_f1 : float
        DPO F1 score.
    save_dir : Path
        Output directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Behavior\nCloning", "Direct Preference\nOptimization"],
        [bc_f1, dpo_f1],
        color=["#6baed6", "#fc8d62"],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, val in zip(bars, [bc_f1, dpo_f1]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2%}",
            ha="center",
            fontweight="bold",
            fontsize=13,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score on Golden Standard Dataset", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "f1_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"F1 comparison saved to {save_dir / 'f1_comparison.png'}")

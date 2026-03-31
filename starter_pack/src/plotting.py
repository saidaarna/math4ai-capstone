"""
Plotting utilities for the Math4AI Capstone project.

This module provides all visualization functions needed for:
  - Decision boundary plots (synthetic tasks)
  - Training dynamics curves (loss / accuracy over epochs)
  - Capacity ablation comparison (hidden widths on moons)
  - Optimizer study comparison (SGD vs Momentum vs Adam)
  - Failure case visualization
  - Advanced Track B calibration figure

All figures are saved to the starter_pack/figures/ directory.
Usage example:
    from src.plotting import plot_decision_boundary, plot_training_curves
    plot_decision_boundary(model, X_test, y_test, "SR on Moons", "figures/moons_sr.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

# Color palette used consistently across all plots
# Class 0 = blue, Class 1 = orange (colorblind-friendly)
CLASS_COLORS  = ["#4878CF", "#D65F5F"]
REGION_COLORS = ["#C9D8F0", "#F5D0D0"]   # lighter fills for decision regions

# Figure resolution for saved files
DPI = 150


# ---------------------------------------------------------------------------
# 1. Decision boundary plot
# ---------------------------------------------------------------------------

def plot_decision_boundary(model, X, y, title, savepath, h=0.02):
    """
    Plots the decision boundary of a trained classifier over a 2D dataset.

    The function works for ANY model that has a .forward(X) method returning
    class probabilities — so it works for both SoftmaxRegression and
    OneHiddenLayerNN without any modification.

    How it works:
      1. Build a fine grid of points covering the data range.
      2. Run the model's forward pass on every grid point.
      3. Take argmax to get the predicted class label for each point.
      4. Color the background by predicted class (decision regions).
      5. Scatter the actual data points on top, colored by true label.

    Parameters
    ----------
    model    : object  — trained model with a .forward(X) method
    X        : ndarray — shape (n, 2), the 2D input features to scatter
    y        : ndarray — shape (n,),  the true integer class labels
    title    : str     — plot title shown above the figure
    savepath : str     — file path where the figure will be saved (e.g. "figures/moons_nn.png")
    h        : float   — grid step size; smaller = finer boundary but slower (default 0.02)
    """

    # --- Build the background grid ---
    # Expand the data range slightly (±0.5) so points near the edges are not clipped
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # np.meshgrid creates a dense 2D grid of (x, y) coordinate pairs
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Flatten the grid into a single matrix of shape (n_grid_points, 2)
    # so we can pass it through the model in one vectorized forward pass
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Run the model on every grid point and take the argmax class prediction
    # Reshape back to the grid shape so it can be plotted with contourf
    Z = model.forward(grid_points).argmax(axis=1).reshape(xx.shape)

    # --- Draw the figure ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # Shade the background regions by predicted class
    # levels=[−0.5, 0.5, 1.5] forces contourf to treat Z as discrete class indices
    num_classes = len(np.unique(y))
    ax.contourf(xx, yy, Z, levels=np.arange(-0.5, num_classes, 1),
                colors=REGION_COLORS[:num_classes], alpha=0.6)

    # Draw a crisp boundary line between regions (no fill, just the contour edge)
    ax.contour(xx, yy, Z, levels=[0.5], colors=["#333333"], linewidths=1.0)

    # Scatter the actual data points colored by their TRUE label
    for cls in range(num_classes):
        mask = y == cls
        ax.scatter(
            X[mask, 0], X[mask, 1],
            color=CLASS_COLORS[cls],
            edgecolors="white",
            linewidths=0.4,
            s=30,
            label=f"Class {cls}",
            zorder=3           # draw on top of the background shading
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot] Decision boundary saved → {savepath}")


# ---------------------------------------------------------------------------
# 2. Training dynamics: loss and accuracy over epochs
# ---------------------------------------------------------------------------

def plot_training_curves(histories, labels, savepath, title="Training Dynamics"):
    """
    Plots train loss, validation loss, and validation accuracy for one or
    more training runs on the same axes — useful for comparing two models
    (e.g. SoftmaxRegression vs OneHiddenLayerNN) on the digits benchmark.

    Parameters
    ----------
    histories : list of dict
        Each dict must have keys 'train_loss', 'val_loss', 'val_acc'
        (lists of floats, one value per epoch) — as returned by train.py.
    labels    : list of str
        Legend labels, one per history dict (e.g. ["Softmax", "NN"]).
    savepath  : str
        File path for the saved figure.
    title     : str
        Overall figure title.
    """

    # We create two subplots side by side:
    #   Left  — cross-entropy loss on train and validation sets
    #   Right — validation accuracy
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Line styles for multiple runs so they are visually distinct in black-and-white too
    line_styles = ["-", "--", "-.", ":"]
    colors      = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]

    for i, (hist, lbl) in enumerate(zip(histories, labels)):
        ls    = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        epochs = range(1, len(hist["train_loss"]) + 1)

        # --- Loss subplot ---
        ax_loss.plot(epochs, hist["train_loss"], ls=ls, color=color,
                     alpha=0.55, linewidth=1.2, label=f"{lbl} — train")
        ax_loss.plot(epochs, hist["val_loss"],   ls=ls, color=color,
                     alpha=1.0,  linewidth=1.8, label=f"{lbl} — val")

        # --- Accuracy subplot ---
        ax_acc.plot(epochs, hist["val_acc"], ls=ls, color=color,
                    linewidth=1.8, label=lbl)

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax_loss.set_title("Loss over epochs", fontsize=11)
    ax_loss.legend(fontsize=8, framealpha=0.8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Validation Accuracy", fontsize=11)
    ax_acc.set_title("Validation accuracy over epochs", fontsize=11)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=9, framealpha=0.8)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot] Training curves saved → {savepath}")


# ---------------------------------------------------------------------------
# 3. Capacity ablation: decision boundaries for multiple hidden widths
# ---------------------------------------------------------------------------

def plot_capacity_ablation(models, widths, X, y, savepath, suptitle="Capacity Ablation: Hidden Width on Moons Task"):
    """
    Shows the effect of hidden-layer width on the learned decision boundary.

    Displays one subplot per model/width in a single row, so the visual
    effect of underfitting (width=2) vs. sufficient capacity (width=32)
    is immediately visible side by side.

    Parameters
    ----------
    models  : list of trained OneHiddenLayerNN objects, one per width
    widths  : list of int, e.g. [2, 8, 32] — the hidden widths used
    X       : ndarray, shape (n, 2)
    y       : ndarray, shape (n,)
    savepath: str
    """

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    fig.suptitle(suptitle,
                 fontsize=13, fontweight="bold")

    # Grid setup — same for every subplot so comparisons are fair
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    num_classes = len(np.unique(y))

    for ax, model, width in zip(axes, models, widths):
        # Decision region background
        Z = model.forward(grid_points).argmax(axis=1).reshape(xx.shape)
        ax.contourf(xx, yy, Z,
                    levels=np.arange(-0.5, num_classes, 1),
                    colors=REGION_COLORS[:num_classes], alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#333333"], linewidths=1.0)

        # Data points
        for cls in range(num_classes):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1],
                       color=CLASS_COLORS[cls], edgecolors="white",
                       linewidths=0.4, s=25, zorder=3)

        # Compute validation accuracy on the provided data so we can annotate
        acc = model.accuracy(X, y)
        ax.set_title(f"Hidden width = {width}\nAcc = {acc:.3f}", fontsize=11)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot] Capacity ablation saved → {savepath}")


# ---------------------------------------------------------------------------
# 4. Optimizer study: val-loss curves for SGD / Momentum / Adam
# ---------------------------------------------------------------------------

def plot_optimizer_study(histories, optimizer_names, savepath):
    """
    Compares training dynamics of different optimizers on the digits benchmark.

    Plots validation cross-entropy loss vs. epoch for each optimizer so the
    convergence speed and final loss level are easy to compare.

    Parameters
    ----------
    histories        : list of dict, each with key 'val_loss' (one per optimizer)
    optimizer_names  : list of str, e.g. ["SGD", "Momentum", "Adam"]
    savepath         : str
    """

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors     = ["#4878CF", "#D65F5F", "#6ACC65"]
    linestyles = ["-", "--", "-."]

    for hist, name, color, ls in zip(histories, optimizer_names, colors, linestyles):
        epochs = range(1, len(hist["val_loss"]) + 1)
        ax.plot(epochs, hist["val_loss"], label=name,
                color=color, linestyle=ls, linewidth=2.0)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
    ax.set_title("Optimizer Study on Digits Benchmark\n(Hidden width = 32, fixed protocol)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot] Optimizer study saved → {savepath}")


# ---------------------------------------------------------------------------
# 5. Failure case visualization
# ---------------------------------------------------------------------------

def plot_failure_case(model, X, y, title, explanation_text, savepath, h=0.02):
    """
    Plots the decision boundary for a deliberately failing model configuration.

    This is intentionally the same as plot_decision_boundary, but it also
    adds an explanation text box in the figure so the failure mechanism is
    immediately visible alongside the visual evidence.

    Parameters
    ----------
    model            : trained (or poorly trained) model with .forward()
    X                : ndarray, shape (n, 2)
    y                : ndarray, shape (n,)
    title            : str, e.g. "Failure: Hidden Width = 2 on Moons"
    explanation_text : str, a short 1–2 sentence explanation of WHY it fails
    savepath         : str
    h                : float, grid resolution
    """

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid_points).argmax(axis=1).reshape(xx.shape)

    num_classes = len(np.unique(y))
    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.contourf(xx, yy, Z,
                levels=np.arange(-0.5, num_classes, 1),
                colors=REGION_COLORS[:num_classes], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors=["#333333"], linewidths=1.0)

    for cls in range(num_classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=CLASS_COLORS[cls], edgecolors="white",
                   linewidths=0.4, s=30, label=f"Class {cls}", zorder=3)

    acc = model.accuracy(X, y)
    ax.set_title(f"{title}\nAccuracy = {acc:.3f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    # Add the explanation as a text box at the bottom of the figure
    # bbox creates a light-yellow rounded box so it stands out from the plot
    ax.text(0.5, -0.18, explanation_text,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                      edgecolor="#CCCCCC", alpha=0.9),
            wrap=True)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot] Failure case saved → {savepath}")


# ---------------------------------------------------------------------------
# 6. Repeated-seed summary bar chart
# ---------------------------------------------------------------------------

def plot_seed_summary(means, cis, model_names, metric_name, savepath):
    """
    Produces a bar chart comparing two models with 95% confidence interval
    error bars — based on the 5-seed repeated evaluation (Section 7.4).

    Parameters
    ----------
    means        : list of float, one mean per model
    cis          : list of float, one half-width CI per model (the ± value)
    model_names  : list of str, e.g. ["Softmax Regression", "1-Hidden-Layer NN"]
    metric_name  : str, e.g. "Test Accuracy" or "Test Cross-Entropy"
    savepath     : str
    """

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    x_positions = np.arange(len(model_names))
    bar_colors  = ["#4878CF", "#D65F5F"]

    bars = ax.bar(x_positions, means,
                  yerr=cis,              # error bars = 95% CI half-widths
                  capsize=8,             # horizontal cap on error bars
                  color=bar_colors,
                  alpha=0.85,
                  edgecolor="white",
                  linewidth=0.8,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    # Annotate each bar with its mean ± CI value
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 0.005,
                f"{mean:.4f} ± {ci:.4f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} — 5-seed comparison\nError bars = 95% CI",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Start y-axis slightly below the smallest bar for visual clarity
    y_min = max(0, min(means) - max(cis) - 0.05)
    ax.set_ylim(bottom=y_min)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot] Seed summary saved → {savepath}")
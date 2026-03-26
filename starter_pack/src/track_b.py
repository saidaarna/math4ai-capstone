"""
Advanced Track B — Prediction Confidence and Reliability Analysis

This module implements Track B of the Math4AI Capstone advanced analysis.
It studies whether a model "knows what it knows" by comparing its expressed
confidence to its actual empirical accuracy on the held-out test set.

Key concepts implemented here:

  Confidence:
    The maximum class probability predicted by the softmax output.
    conf(x) = max_j P(Y=j | x)
    A confidence of 0.95 means the model is 95% sure of its prediction.
    If the model is well-calibrated, a confidence of 0.95 should correspond
    to being correct roughly 95% of the time.

  Predictive Entropy:
    A measure of uncertainty across the full predicted distribution.
    H(x) = -sum_j [ p_j * log(p_j) ]
    High entropy → the model spreads probability mass across many classes.
    Low entropy  → the model concentrates mass on one class (high confidence).
    For k classes, max entropy = log(k), achieved when all classes are equally likely.

  Calibration:
    A model is well-calibrated if its confidence score matches the actual
    fraction of correct predictions (empirical accuracy).
    Example: among all predictions made with ~70% confidence, the model
    should be correct roughly 70% of the time.
    We check this via a 5-bin reliability table (Section 8.2 of the handout).

Usage example:
    from src.track_b import run_track_b
    run_track_b(sr_model, nn_model, X_test, y_test, figures_dir="figures/")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Shared style constants (mirrors plotting.py for visual consistency)
# ---------------------------------------------------------------------------

DPI          = 150
SR_COLOR     = "#4878CF"   # blue  — Softmax Regression
NN_COLOR     = "#D65F5F"   # red   — Neural Network
PERFECT_COLOR = "#888888"  # gray  — perfect-calibration reference line


# ---------------------------------------------------------------------------
# 1. Core metric computation
# ---------------------------------------------------------------------------

def compute_confidence_metrics(model, X, y):
    """
    Runs a forward pass and extracts per-sample confidence and entropy.

    Parameters
    ----------
    model : trained model with a .forward(X) method returning
            predicted probabilities P of shape (n, k)
    X     : ndarray, shape (n, d) — input features
    y     : ndarray, shape (n,)  — true integer class labels

    Returns
    -------
    conf    : ndarray, shape (n,) — max predicted probability per sample
    entropy : ndarray, shape (n,) — predictive entropy per sample
    correct : ndarray, shape (n,) — bool array, True where prediction == true label
    P       : ndarray, shape (n, k) — full predicted probability matrix
    """

    # Run the forward pass to get class probabilities for every test sample
    P = model.forward(X)           # shape: (n, k)

    # --- Confidence ---
    # The predicted class is the one with the highest probability (argmax).
    # The confidence for that prediction is the value of that max probability.
    conf = P.max(axis=1)           # shape: (n,) — one confidence score per sample

    # --- Predictive Entropy ---
    # H = -sum_j [ p_j * log(p_j) ]
    # We add a tiny epsilon (1e-15) to avoid log(0) which would produce -inf.
    entropy = -(P * np.log(P + 1e-15)).sum(axis=1)   # shape: (n,)

    # --- Correctness ---
    # A prediction is correct when the argmax class matches the true label.
    preds   = P.argmax(axis=1)     # shape: (n,) — predicted class indices
    correct = (preds == y)         # shape: (n,) — boolean mask

    return conf, entropy, correct, P


# ---------------------------------------------------------------------------
# 2. Five-bin calibration table
# ---------------------------------------------------------------------------

def calibration_table(conf, correct, n_bins=5):
    """
    Builds a reliability table by binning test samples by their confidence
    and measuring empirical accuracy within each bin.

    For a well-calibrated model, mean confidence ≈ empirical accuracy
    in every bin. A large gap reveals overconfidence or underconfidence.

    Parameters
    ----------
    conf    : ndarray, shape (n,) — max predicted probability
    correct : ndarray, shape (n,) — boolean correctness mask
    n_bins  : int — number of equal-width bins between 0 and 1 (default 5)

    Returns
    -------
    table : list of dicts, one per bin, with keys:
        'bin_range'   — (lower, upper) confidence boundaries of the bin
        'n_samples'   — number of test samples in this bin
        'mean_conf'   — average confidence of samples in this bin
        'empirical_acc' — fraction of samples in this bin that were correct
        'gap'         — mean_conf minus empirical_acc (positive = overconfident)
    """

    # Create n_bins equal-width intervals spanning [0, 1]
    # e.g. n_bins=5 → [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    table = []

    for i in range(n_bins):
        low  = bin_edges[i]
        high = bin_edges[i + 1]

        # Select all samples whose confidence falls within this bin
        # For the final bin we use <= to include confidence = 1.0 exactly
        if i < n_bins - 1:
            mask = (conf >= low) & (conf < high)
        else:
            mask = (conf >= low) & (conf <= high)

        n_in_bin = mask.sum()

        if n_in_bin == 0:
            # Skip empty bins — they carry no information
            table.append({
                "bin_range":    (round(low, 2), round(high, 2)),
                "n_samples":    0,
                "mean_conf":    None,
                "empirical_acc": None,
                "gap":          None,
            })
            continue

        mean_conf    = conf[mask].mean()
        empirical_acc = correct[mask].mean()

        # Gap > 0 means the model is OVERCONFIDENT in this bin
        # Gap < 0 means the model is UNDERCONFIDENT in this bin
        gap = mean_conf - empirical_acc

        table.append({
            "bin_range":     (round(low, 2), round(high, 2)),
            "n_samples":     int(n_in_bin),
            "mean_conf":     float(mean_conf),
            "empirical_acc": float(empirical_acc),
            "gap":           float(gap),
        })

    return table


def print_calibration_table(table, model_name):
    """
    Prints the calibration table in a clean, readable format for the terminal.

    Parameters
    ----------
    table      : list of dicts, as returned by calibration_table()
    model_name : str, e.g. "Softmax Regression" or "1-Hidden-Layer NN"
    """

    print(f"\n{'=' * 60}")
    print(f"Calibration Table — {model_name}")
    print(f"{'=' * 60}")
    header = f"{'Bin Range':<16} {'N':>6} {'Mean Conf':>11} {'Emp. Acc':>10} {'Gap':>8}"
    print(header)
    print("-" * 60)

    for row in table:
        low, high = row["bin_range"]

        if row["n_samples"] == 0:
            print(f"[{low:.2f}, {high:.2f})   {'0':>6}   {'—':>11}   {'—':>10}   {'—':>8}")
            continue

        gap_str = f"{row['gap']:+.4f}"   # '+' prefix shows sign explicitly

        print(
            f"[{low:.2f}, {high:.2f})   "
            f"{row['n_samples']:>6}   "
            f"{row['mean_conf']:>11.4f}   "
            f"{row['empirical_acc']:>10.4f}   "
            f"{gap_str:>8}"
        )

    print("=" * 60)


# ---------------------------------------------------------------------------
# 3. Confidence vs empirical accuracy: reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(table_sr, table_nn, savepath):
    """
    Plots the reliability (calibration) diagram for both models side by side.

    For a perfectly calibrated model, the plotted line would sit exactly on
    the diagonal y = x.  Deviations above the diagonal mean the model is
    UNDERCONFIDENT (it's more accurate than it thinks).
    Deviations below mean OVERCONFIDENCE (it's less accurate than it claims).

    Parameters
    ----------
    table_sr : list of dicts — calibration table for Softmax Regression
    table_nn : list of dicts — calibration table for Neural Network
    savepath : str
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Reliability Diagrams (Calibration)\n"
                 "Diagonal = perfect calibration",
                 fontsize=13, fontweight="bold")

    for ax, table, name, color in zip(
        axes,
        [table_sr, table_nn],
        ["Softmax Regression", "1-Hidden-Layer NN"],
        [SR_COLOR, NN_COLOR]
    ):
        # Keep only non-empty bins for plotting
        valid = [row for row in table if row["n_samples"] > 0]

        mean_confs = [row["mean_conf"]    for row in valid]
        emp_accs   = [row["empirical_acc"] for row in valid]
        n_samples  = [row["n_samples"]     for row in valid]

        # --- Perfect calibration reference diagonal ---
        ax.plot([0, 1], [0, 1], linestyle="--", color=PERFECT_COLOR,
                linewidth=1.5, label="Perfect calibration")

        # --- Model calibration line ---
        # Use scatter where marker size encodes the number of samples in each bin
        # This makes it visually clear which bins are most important
        norm_sizes = [50 + 150 * (n / max(n_samples)) for n in n_samples]
        ax.scatter(mean_confs, emp_accs,
                   s=norm_sizes, color=color, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.6,
                   label=f"{name}")
        ax.plot(mean_confs, emp_accs, color=color, linewidth=1.5, alpha=0.6)

        # Shade the gap between perfect calibration and actual calibration
        ax.fill_between(mean_confs, mean_confs, emp_accs,
                        alpha=0.08, color=color, label="Calibration gap")

        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean Confidence (max p)", fontsize=11)
        ax.set_ylabel("Empirical Accuracy", fontsize=11)
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # Annotate each point with its sample count
        for mc, ea, n in zip(mean_confs, emp_accs, n_samples):
            ax.annotate(f"n={n}", xy=(mc, ea),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="#555555")

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[track_b] Reliability diagram saved → {savepath}")


# ---------------------------------------------------------------------------
# 4. Confidence distribution: correct vs incorrect predictions
# ---------------------------------------------------------------------------

def plot_confidence_by_correctness(conf_sr, correct_sr,
                                   conf_nn, correct_nn,
                                   savepath):
    """
    Shows histograms of model confidence separately for correctly classified
    and incorrectly classified test samples.

    A well-behaved model should:
      — show high confidence when it is correct (histogram concentrated near 1)
      — show lower confidence when it is wrong   (histogram spread or near 1/k)

    If the 'wrong' histogram is also concentrated near 1, the model is
    overconfident on its mistakes — an important failure mode to discuss.

    Parameters
    ----------
    conf_sr   : ndarray, shape (n,) — confidence scores for Softmax Regression
    correct_sr: ndarray, shape (n,) — boolean correctness for Softmax Regression
    conf_nn   : ndarray, shape (n,) — confidence scores for Neural Network
    correct_nn: ndarray, shape (n,) — boolean correctness for Neural Network
    savepath  : str
    """

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fig.suptitle("Confidence Distribution: Correct vs Incorrect Predictions",
                 fontsize=13, fontweight="bold")

    bins = np.linspace(0, 1, 21)   # 20 bins across the [0, 1] confidence range

    configs = [
        (axes[0, 0], conf_sr[correct_sr],  SR_COLOR, "Softmax — Correct",   True),
        (axes[0, 1], conf_sr[~correct_sr], SR_COLOR, "Softmax — Incorrect",  False),
        (axes[1, 0], conf_nn[correct_nn],  NN_COLOR, "NN — Correct",         True),
        (axes[1, 1], conf_nn[~correct_nn], NN_COLOR, "NN — Incorrect",       False),
    ]

    for ax, data, color, title, is_correct in configs:
        ax.hist(data, bins=bins, color=color,
                alpha=0.75, edgecolor="white", linewidth=0.4)

        # Add a vertical dashed line at the median confidence
        if len(data) > 0:
            med = np.median(data)
            ax.axvline(med, color="#333333", linestyle="--", linewidth=1.2,
                       label=f"Median = {med:.3f}")
            ax.legend(fontsize=9)

        n_label = f"n = {len(data)}"
        ax.text(0.04, 0.93, n_label,
                transform=ax.transAxes, fontsize=9, color="#555555",
                va="top")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Confidence (max predicted probability)", fontsize=10)
        ax.set_ylabel("Number of test samples", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[track_b] Confidence-by-correctness plot saved → {savepath}")


# ---------------------------------------------------------------------------
# 5. Entropy comparison: correct vs incorrect, side by side
# ---------------------------------------------------------------------------

def plot_entropy_comparison(entropy_sr, correct_sr,
                             entropy_nn, correct_nn,
                             savepath):
    """
    Plots box plots of predictive entropy for correct vs incorrect predictions
    for both models, on a single figure.

    Entropy is higher when the model is more uncertain (probability spread
    across many classes).  We expect:
      - Correct predictions  → low entropy  (confident and right)
      - Incorrect predictions → higher entropy (uncertain or overconfident)

    Parameters
    ----------
    entropy_sr : ndarray — predictive entropy for Softmax Regression
    correct_sr : ndarray — boolean correctness for Softmax Regression
    entropy_nn : ndarray — predictive entropy for Neural Network
    correct_nn : ndarray — boolean correctness for Neural Network
    savepath   : str
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Predictive Entropy: Correct vs Incorrect Predictions",
                 fontsize=13, fontweight="bold")

    for ax, entropy, correct, name, color in zip(
        axes,
        [entropy_sr, entropy_nn],
        [correct_sr, correct_nn],
        ["Softmax Regression", "1-Hidden-Layer NN"],
        [SR_COLOR, NN_COLOR]
    ):
        # Separate entropy values into two groups: correct and incorrect
        ent_correct   = entropy[correct]
        ent_incorrect = entropy[~correct]

        # Box plot: the box spans Q1–Q3, the line is the median,
        # whiskers extend to 1.5 × IQR, and dots are outliers
        bp = ax.boxplot(
            [ent_correct, ent_incorrect],
            labels=["Correct", "Incorrect"],
            patch_artist=True,         # filled boxes instead of outline-only
            medianprops={"color": "#333333", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4}
        )

        # Color the boxes: correct = lighter, incorrect = the model's main color
        bp["boxes"][0].set_facecolor(color + "55")   # ~33% opacity
        bp["boxes"][1].set_facecolor(color + "CC")   # ~80% opacity

        # Annotate with mean entropy values for quick comparison
        for i, group in enumerate([ent_correct, ent_incorrect], start=1):
            if len(group) > 0:
                ax.text(i, ax.get_ylim()[1] * 0.95,
                        f"μ={group.mean():.3f}",
                        ha="center", va="top", fontsize=9, color="#444444")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Prediction outcome", fontsize=10)
        ax.set_ylabel("Predictive entropy H(x)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[track_b] Entropy comparison saved → {savepath}")


# ---------------------------------------------------------------------------
# 6. Master function — run the entire Track B pipeline
# ---------------------------------------------------------------------------

def run_track_b(sr_model, nn_model, X_test, y_test, figures_dir="figures/"):
    """
    Runs the complete Track B analysis pipeline and saves all required
    figures and tables to figures_dir.

    Call this once after both models have been trained and checkpointed.

    Parameters
    ----------
    sr_model    : trained SoftmaxRegression instance
    nn_model    : trained OneHiddenLayerNN instance
    X_test      : ndarray, shape (n_test, d)
    y_test      : ndarray, shape (n_test,)
    figures_dir : str — directory where figures will be saved (must exist)

    Returns
    -------
    results : dict — summary statistics for both models, suitable for the report
    """

    import os
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Running Track B — Confidence and Reliability Analysis")
    print("=" * 60)

    # --- Step 1: Compute confidence and entropy for both models ---
    conf_sr, entropy_sr, correct_sr, P_sr = compute_confidence_metrics(sr_model, X_test, y_test)
    conf_nn, entropy_nn, correct_nn, P_nn = compute_confidence_metrics(nn_model, X_test, y_test)

    print(f"\nTest set size : {len(y_test)} samples")
    print(f"\nSoftmax Regression —  accuracy : {correct_sr.mean():.4f}"
          f"  |  mean confidence : {conf_sr.mean():.4f}"
          f"  |  mean entropy : {entropy_sr.mean():.4f}")
    print(f"Neural Network     —  accuracy : {correct_nn.mean():.4f}"
          f"  |  mean confidence : {conf_nn.mean():.4f}"
          f"  |  mean entropy : {entropy_nn.mean():.4f}")

    # --- Step 2: Build 5-bin calibration tables ---
    table_sr = calibration_table(conf_sr, correct_sr, n_bins=5)
    table_nn = calibration_table(conf_nn, correct_nn, n_bins=5)

    print_calibration_table(table_sr, "Softmax Regression")
    print_calibration_table(table_nn, "1-Hidden-Layer NN")

    # --- Step 3: Reliability diagram (Figure for the report) ---
    plot_reliability_diagram(
        table_sr, table_nn,
        savepath=f"{figures_dir}/track_b_reliability_diagram.png"
    )

    # --- Step 4: Confidence histogram: correct vs incorrect ---
    plot_confidence_by_correctness(
        conf_sr, correct_sr,
        conf_nn, correct_nn,
        savepath=f"{figures_dir}/track_b_confidence_distribution.png"
    )

    # --- Step 5: Entropy comparison box plot ---
    plot_entropy_comparison(
        entropy_sr, correct_sr,
        entropy_nn, correct_nn,
        savepath=f"{figures_dir}/track_b_entropy_comparison.png"
    )

    # --- Step 6: Return a summary dict for use in the report write-up ---
    results = {
        "softmax": {
            "accuracy":        correct_sr.mean(),
            "mean_confidence": conf_sr.mean(),
            "mean_entropy":    entropy_sr.mean(),
            "calibration_table": table_sr,
        },
        "neural_net": {
            "accuracy":        correct_nn.mean(),
            "mean_confidence": conf_nn.mean(),
            "mean_entropy":    entropy_nn.mean(),
            "calibration_table": table_nn,
        },
    }

    print("\n[track_b] All Track B figures saved successfully.")
    return results
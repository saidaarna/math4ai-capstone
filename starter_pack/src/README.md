# `src/`

This directory contains the full from-scratch implementation of the Math4AI Capstone project.
All core learning algorithms are implemented in NumPy only. PyTorch, TensorFlow, JAX and
scikit-learn model classes are not used anywhere in this directory.

---

## File Overview

### data_loader.py

Loads all three datasets used in the project. File paths are resolved dynamically relative
to the script's location, so the loader works regardless of the working directory the user
runs their script from.

- `load_synthetic(file_path)` - loads `linear_gaussian.npz` or `moons.npz`, which come
  pre-split into train, validation and test sets.
- `load_digits(data_path, split_path)` - loads `digits_data.npz` and applies the fixed
  index split from `digits_split_indices.npz`. Returns three `(X, y)` tuples in order:
  train, validation, test.

---

### softmax_regression.py

Implements the linear baseline model: multiclass softmax regression with L2 regularization.

- `__init__(d, k, lam, seed)` - initializes weights `W` of shape `(k, d)` scaled by 0.01
  and zero biases `b` of shape `(k,)`.
- `forward(X)` - computes class probabilities via the affine map `S = XW^T + b` followed
  by numerically stable softmax (row-wise max subtraction before exponentiation).
- `loss(X, y)` - returns mean cross-entropy loss plus the L2 regularization term
  `(lambda / 2) * ||W||^2`.
- `gradients(X, y)` - returns analytical gradients `dW` and `db` via the chain rule.
  The gradient of the softmax cross-entropy with respect to the logits is `(P - Y) / n`.
  The weight gradient includes the L2 penalty term `lambda * W`.
- `accuracy(X, y)` - returns the fraction of correctly classified examples.

---

### neural_network.py

Implements the one-hidden-layer neural network with tanh activation and softmax output.

- `__init__(d, hidden, k, lam, seed)` - initializes `W1` of shape `(hidden, d)` and `W2`
  of shape `(k, hidden)`, both scaled by 0.1, with zero biases `b1` and `b2`.
- `forward(X)` - forward pass: `Z1 = XW1^T + b1`, `H = tanh(Z1)`, `S = HW2^T + b2`,
  `P = softmax(S)`. Intermediate values are stored on `self` for use in backpropagation.
- `loss(X, y)` - cross-entropy plus L2 regularization over both weight matrices
  `W1` and `W2`.
- `gradients(X, y)` - full vectorized backpropagation. Gradients are computed in reverse
  order: output layer first (`dS`, `dW2`, `db2`), then back through `W2` to the hidden
  layer (`dH`, `dZ1` using the tanh derivative `1 - H^2`), then `dW1` and `db1`.
  Both weight gradients include the L2 penalty.
- `accuracy(X, y)` - same zero-one accuracy as the softmax model.

---

### optimizers.py

Implements three parameter update rules used in the optimizer study (Section 7.3).

All optimizers share the same interface: `update(params, grads)` takes a list of parameter
arrays and a list of gradient arrays and returns a list of updated parameter arrays.
This uniform interface means the training loop does not need to know which optimizer is active.

- `SGD(lr)` - standard gradient descent: `p = p - lr * g`.
- `Momentum(lr, momentum)` - accumulates a velocity vector `v` and applies
  `v = mu * v - lr * g`, then `p = p + v`. Reduces oscillation and speeds up convergence
  along consistent gradient directions.
- `Adam(lr, b1, b2, eps)` - maintains per-parameter first and second moment estimates
  with bias correction. Default hyperparameters follow the capstone protocol:
  `lr=0.001`, `b1=0.9`, `b2=0.999`, `eps=1e-8`.

---

### train.py

Mini-batch training loops with validation checkpointing.

- `train_softmax(model, optimizer, X_tr, y_tr, X_val, y_val, epochs, batch_size)` -
  trains a `SoftmaxRegression` model. Shuffles training data each epoch, iterates through
  mini-batches of size 64 and saves the parameter state whenever validation loss improves.
  Restores the best checkpoint before returning.
- `train_nn(model, optimizer, X_tr, y_tr, X_val, y_val, epochs, batch_size)` -
  identical logic for `OneHiddenLayerNN`, handling all four parameter arrays
  (`W1`, `b1`, `W2`, `b2`).

Both functions return a `history` dictionary with keys `train_loss`, `val_loss` and
`val_acc`, one value per epoch, for use in training dynamics plots.

---

### evaluate.py

- `evaluate(model, X, y)` - computes classification accuracy and unregularized mean
  cross-entropy loss. The regularization term is deliberately excluded here because the
  capstone protocol specifies reporting the average negative log-probability assigned to
  the true class, not the training objective. Use this function for all validation and
  test set reporting.

---

### plotting.py

All visualization functions used in the report and experiments. Every function saves its
figure to disk and prints the save path to stdout.

- `plot_decision_boundary(model, X, y, title, savepath)` - builds a fine 2D grid, runs
  the model's forward pass on every point and plots the resulting decision regions with
  the true data points overlaid. Works with any model that has a `forward(X)` method.
- `plot_training_curves(histories, labels, savepath)` - plots train loss, validation loss,
  and validation accuracy over epochs. Accepts a list of history dicts so multiple models
  can be compared on a single canvas.
- `plot_capacity_ablation(models, widths, X, y, savepath)` - one subplot per hidden width,
  showing the decision boundary and test accuracy for each capacity setting.
- `plot_optimizer_study(histories, optimizer_names, savepath)` - validation loss curves
  for all three optimizers on the same axes.
- `plot_failure_case(model, X, y, title, explanation_text, savepath)` - decision boundary
  plot with an explanation text box embedded below the axes, documenting the failure mechanism.
- `plot_seed_summary(means, cis, model_names, metric_name, savepath)` - bar chart with
  95% confidence interval error bars for the repeated-seed evaluation results.

---

### track_b.py

Implements Advanced Track B: prediction confidence and reliability analysis.

- `compute_confidence_metrics(model, X, y)` - returns per-sample confidence (max predicted
  probability), predictive entropy `H = -sum p_j log p_j` and a boolean correctness mask.
- `calibration_table(conf, correct, n_bins)` - divides test samples into bins by confidence
  level and computes the empirical accuracy within each bin. Returns a list of dicts with
  `bin_range`, `n_samples`, `mean_conf`, `empirical_acc` and `gap` (positive gap means
  the model is overconfident).
- `print_calibration_table(table, model_name)` - prints the calibration table to stdout
  in a readable format.
- `plot_reliability_diagram(table_sr, table_nn, savepath)` - reliability diagram for both
  models with the perfect-calibration diagonal as a reference.
- `plot_confidence_by_correctness(...)` - histograms of confidence scores split by whether
  the prediction was correct or incorrect.
- `plot_entropy_comparison(...)` - box plots of predictive entropy for correct versus
  incorrect predictions.
- `run_track_b(sr_model, nn_model, X_test, y_test, figures_dir)` - master function that
  runs the complete Track B pipeline and saves all figures. Call this once after both
  models have been trained.

---

### ablations_and_stats.py

Runs the required ablation studies (Section 7.3) and repeated-seed statistical evaluation
(Section 7.4).

- `run_ablation_studies()` - trains the neural network with hidden widths `{2, 8, 32}` on
  the moons task and saves decision boundary plots. Then trains the neural network with
  SGD, Momentum and Adam on the digits benchmark and saves the optimizer comparison plot.
- `run_statistical_analysis()` - runs both models across five seeds `{0, 1, 2, 3, 4}` on
  the digits benchmark. Evaluates each trained model on the test set and computes the
  mean and 95% confidence interval for accuracy and cross-entropy using the formula
  `mean +/- 2.776 * (std / sqrt(5))`.

---

### experiments.py

Runs the three required core comparisons (Section 7.1).

- Experiment 1: both models on the linear Gaussian task. Expected result: similar
  performance, because the true decision boundary is linear and softmax regression
  already expresses it.
- Experiment 2: both models on the moons task. Expected result: the neural network
  substantially outperforms softmax regression because the boundary is nonlinear.
- Experiment 3: both models on the digits benchmark with the fixed split and protocol.
  Reports accuracy and cross-entropy on train, validation and test sets for both models,
  and saves the training dynamics figure.

---

### sanity_checks.py

Documents implementation correctness with four concrete checks, as required in Section 5.5
of the handout.

- Check 1: predicted class probabilities sum to 1.0 for every sample.
- Check 2: loss decreases on a five-example subset after 200 gradient steps.
- Check 3: analytical gradient matches the numerical finite-difference estimate to within
  a relative error of `1e-4`.
- Check 4: loss is finite (no NaN or Inf) after a forward pass.

Run `python sanity_checks.py` from this directory to execute all checks.

---

### inspect_data.ipynb

A Jupyter notebook used during Day 1 to inspect the shapes, class distributions and value
ranges of all four data files before any model was implemented. Not part of the model
pipeline; included for transparency and reproducibility of the data inspection step.

---

## Running the Code

All scripts are designed to be run from inside the `src/` directory:

```
cd starter_pack/src
python sanity_checks.py
python experiments.py
python ablations_and_stats.py
```

Track B is called from within a script after models are trained:

```python
from track_b import run_track_b
run_track_b(sr_model, nn_model, X_test, y_test, figures_dir="../figures/")
```

Figures are saved to `starter_pack/figures/`. That directory must exist before running
the experiment scripts, or it will be created automatically by `ablations_and_stats.py`.

---

## Dependencies

```
python >= 3.9
numpy
matplotlib
```

No other packages are required or permitted for the core model implementations.


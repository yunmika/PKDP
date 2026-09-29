# Command Line Reference

PKDP provides two primary subcommands:
- `python PKDP.py train`: Trains a genomic prediction model with optional hyperparameter optimization.
- `python PKDP.py predict`: Evaluates or predicts phenotypes on test samples using a trained model checkpoint.

---

## 1. `PKDP.py train`

### Required Arguments
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--train_phe` | `str` | Path to the training phenotype CSV file. |
| `--geno` | `str` | Path to the full-genome genotype CSV file. |
| `--output_path` | `str` | Output directory where trained model checkpoints and training logs are saved. |

### Training & Tuning Arguments
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prefix` | `str` | Timestamp | Prefix for output model and result files (e.g., `demo3`). |
| `--pnum` | `str`/`int` | `None` (1st column) | Target phenotype column name or 0-indexed column index. |
| `--optuna_trials` | `int` | `50` | Number of Optuna Bayesian trials for learning rate tuning (each trial runs 5-Fold CV). **Set to `0` to skip search and train instantly in 1~2 seconds.** |
| `--epochs` | `int` | `100` | Maximum number of training epochs. |
| `--batch_size` | `int` | `32` | Mini-batch size. |
| `--optimizer` | `str` | `AdamW` | Optimizer choice: `AdamW`, `Adam`, or `SGD`. |
| `--device` | `str` | `cuda` | Target compute device: `cuda` (or `cuda:0`) or `cpu`. |
| `--early_stop` | flag | `True` | Enables early stopping when loss plateaus for 10 consecutive epochs. |
| `--seed` | `int` | `None` | Random seed for data shuffling and weight initialization. |
| `--adjust_encoding`| flag | `False` | Adjusts genotype values from $\{0, 1, 2\}$ to $\{-1, 0, 1\}$. |
| `--cv_folds` | `int` | `0` | Extra K-fold cross-validation during final training (0 = train single full model). |

### Prior Knowledge Arguments
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prior_features_file` | `str` | `None` | Path to a text file with one prior feature ID per line. *(Recommended)* |
| `--prior_features` | `str ...` | `None` | Space-separated list of marker IDs or column indices (e.g., `"10 20"`). |

### Architecture Arguments
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--conv_kernel_size` | `int ...`| `5 11 21` | Parallel multi-scale 1D convolution kernel sizes for the polygenic main path. |
| `--main_channels` | `int ...`| `64 32 32`| Number of feature channels across the 3 convolutional residual blocks. |
| `--fc_units` | `int ...`| `128 64` | Hidden dimensions in fully connected prediction head. |
| `--dropout` | `float` | `0.2` | Dropout rate for regularization in FC layers. |

---

## 2. `PKDP.py predict`

### Required Arguments
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--geno` | `str` | Path to test genotype CSV file. |
| `--model_path` | `str` | Path to the trained `.pth` model file (saved from `train`). |
| `--output_path` | `str` | Directory to save prediction outputs and visualization plots. |

### Optional Arguments
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--test_phe` | `str` | `None` | Path to testing phenotype CSV (optional; if provided, evaluation metrics and scatter plots are generated). |
| `--prefix` | `str` | Timestamp | Prefix for prediction output files. |
| `--pnum` | `str` | `None` (1st column) | Target phenotype column name to evaluate against. |
| `--device` | `str` | `cuda` | Target compute device: `cuda` or `cpu`. |
| `--adjust_encoding`| flag | `False` | Adjusts genotype values from $\{0, 1, 2\}$ to $\{-1, 0, 1\}$ (should match training setting). |
| `--prior_features_file` | `str` | `None` | Prior features file (optional; automatically aligns with model). |

### Output Files
When `--test_phe` is supplied, `predict` automatically exports:
1. `*_predictions.csv`: Table containing sample IDs, observed values, and predicted GEBV.
2. `*_metrics.csv`: Metric summary including Pearson Correlation, Spearman, $R^2$, RMSE, and MAE.
3. `*_combined_results.png`: High-resolution dual plot displaying density distribution and predicted vs. observed scatter plot.
4. `*_residuals.png`: Residual density and error distribution plot.

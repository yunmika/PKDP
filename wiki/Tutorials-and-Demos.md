# Tutorials & Benchmark Demos

PKDP includes 3 simulated benchmark datasets located in `demo/`, representing three classical genetic architectures in quantitative genetics.

---

## 1. Demo Datasets Overview

| Dataset | Genetic Architecture | Features ($P$) | Prior QTLs ($K$) | Training Samples ($N_{\text{tr}}$) | Testing Samples ($N_{\text{te}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Demo 1** | Additive + LD | 3,000 | 20 | 700 | 300 |
| **Demo 2** | Additive + Dominance + LD | 3,000 | 20 | 700 | 300 |
| **Demo 3** | Additive + Epistasis ($A \times A$) + LD | 3,000 | 20 | 700 | 300 |

---

## 2. Step-by-Step Walkthrough

### Step 1: Model Training
Train a PKDP model for Demo 3 using the GPU:

```bash
# Standard training with hyperparameter search (default 50 trials)
python PKDP.py train \
    --train_phe demo/demo3_train_phe.csv \
    --geno demo/demo3_train_geno.csv \
    --prior_features_file demo/demo3_prior_features.txt \
    --output_path results_demo3/ \
    --prefix demo3 \
    --epochs 100 \
    --device cuda

# Fast execution (skipping Optuna search, runs in ~2 seconds)
python PKDP.py train \
    --train_phe demo/demo3_train_phe.csv \
    --geno demo/demo3_train_geno.csv \
    --prior_features_file demo/demo3_prior_features.txt \
    --output_path results_demo3/ \
    --prefix demo3 \
    --optuna_trials 0 \
    --epochs 50 \
    --device cuda
```

### Step 2: Phenotype Prediction & Evaluation
Evaluate the trained model on independent testing samples:

```bash
python PKDP.py predict \
    --geno demo/demo3_test_geno.csv \
    --test_phe demo/demo3_test_phe.csv \
    --prior_features_file demo/demo3_prior_features.txt \
    --model_path results_demo3/demo3_best_model.pth \
    --output_path predictions_demo3/ \
    --prefix demo3_pred \
    --device cuda
```

Outputs are automatically generated under `predictions_demo3/`:
- `demo3_pred_predictions.csv`
- `demo3_pred_metrics.csv`
- `demo3_pred_combined_results.png`

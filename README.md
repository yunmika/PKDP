<p  align="center"><img src="https://github.com/user-attachments/assets/f5217d49-3385-44fa-9776-8f398829c1fb" style="width: 50%; height: auto;">

<div align="center">

**Prior Knowledge Dual-Path CNN**

</div>

***

<div align="center">

[![Release Version](https://img.shields.io/github/v/release/yunmika/PKDP?color=blue)](https://github.com/yunmika/PKDP/releases)
[![License](https://img.shields.io/github/license/yunmika/PKDP?color=green)](https://github.com/yunmika/PKDP/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/yunmika/PKDP?color=orange)](https://github.com/yunmika/PKDP/commits/main)

</div>

PKDP (Prior Knowledge Dual-Path CNN) is a dual-path convolutional neural network framework designed to enhance genomic selection (GS) by integrating genome-wide association study (GWAS) results with genome-wide minor-effect markers.

<p  align="center"><img src="https://github.com/user-attachments/assets/089f554e-3743-49a8-9672-61dda37afaa2" style="width: 70%; height: auto;">

---

## Installation

```bash
# Create a new conda environment
conda create -n PKDP_env python=3.8

# Activate the environment (works on all platforms)
conda activate PKDP_env

# Install PKDP
git clone https://github.com/yunmika/PKDP.git
cd ./PKDP

# Install dependencies
pip install -r requirements.txt
```

## Requirement
- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0
- SciPy >= 1.7.0
- Optuna >= 2.10.0


## Options and usage

### Training

```sh
python ./PKDP/PKDP.py --mode train -h
```

#### Required Parameters
| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `--mode train`    | Specify training mode                           |
| `--train_phe`     | Path to the training phenotype file             |
| `--geno`          | Path to the genotype file                       |
| `--output_path`   | Directory to save outputs                       |

#### Optional Parameters
| Parameter                | Description                                      | Default Value |
|--------------------------|--------------------------------------------------|---------------|
| `--test_phe`             | Path to the testing phenotype file              | None          |
| `--pnum`                 | Phenotype column index or name                 | First column  |
| `--prefix`               | Prefix for output files                         | Timestamp     |
| `--batch_size`           | Batch size for training                         | 32            |
| `--epochs`               | Number of training epochs                       | 50            |
| `--optuna_trials`        | Number of Optuna trials for hyperparameter tuning | 20          |
| `--device`               | Device to use (`cuda` or `cpu`)                 | cuda:0        |
| `--optimizer`            | Optimizer type (`Adam`, `SGD`, `AdamW`)         | Adam          |
| `--early_stop`           | Enable early stopping                           | False         |
| `--prior_features`       | Prior knowledge features (space-separated IDs)  | None          |
| `--prior_features_file`  | Path to a file with one prior feature ID per line | None        |
| `--adjust_encoding`      | Adjust genotype encoding from {0,1,2} to {-1,0,1} | False      |


#### Model Architecture Parameters
| Parameter            | Description                                      | Default Value |
|----------------------|--------------------------------------------------|---------------|
| `--conv_kernel_size` | Kernel sizes for main convolution path          | [11]          |
| `--prior_kernel_size`| Kernel sizes for prior path convolution         | [5]           |
| `--main_channels`    | Number of channels in main convolution path     | [64, 32, 32]  |
| `--prior_channels`   | Number of channels in prior knowledge path      | [16, 16, 32]  |
| `--fc_units`         | Number of units in fully connected layers       | [128, 64]     |
| `--dropout`          | Dropout probability for fully connected layers  | 0             |

### Prediction

```sh
python ./PKDP/PKDP.py --mode predict -h
```

#### Required Parameters
| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `--mode predict`  | Specify prediction mode                         |
| `--geno`          | Path to the genotype file                       |
| `--model_path`    | Path to the trained model file                  |
| `--output_path`   | Directory to save outputs                       |

#### Optional Parameters
| Parameter                | Description                                      | Default Value |
|--------------------------|--------------------------------------------------|---------------|
| `--test_phe`             | Path to the testing phenotype file (optional)   | None          |
| `--pnum`                 | Phenotype column index or name                 | First column  |
| `--prefix`               | Prefix for output files                         | Timestamp     |
| `--device`               | Device to use (`cuda` or `cpu`)                 | cuda:0        |
| `--adjust_encoding`      | Adjust genotype encoding from {0,1,2} to {-1,0,1} | False      |
| `--prior_features`       | Prior knowledge features (space-separated IDs)  | None          |
| `--prior_features_file`  | Path to a file with one prior feature ID per line | None        |


#### note
- Input phenotype data format: see `./demo/demo_phenotypes.csv`
- Input genotype data format: see `./demo/demo_genotypes.csv`
- During model training, please pay close attention to adjusting the following hyperparameters, as they significantly impact model performance:
    *   `--main_channels`: Number of channels in the main network.
    *   `--prior_channels`: Number of channels in the prior knowledge network.
    *   `--learning_rate`: Learning rate.
    *   `--batch_size`: Batch size.

## Usage Examples

### Training

To train the model with cross-validation:

```bash
python PKDP.py --mode train --train_phe data/train_phe.csv --geno data/geno.csv \
               --test_phe data/test_phe.csv --output_path results/ \
               --prior_features "SNP1 SNP2 SNP3" --main_channels 64 32 32 --prior_channels 16 32 32
```

### Prediction

To make predictions using a trained model with phenotype data for evaluation:


```bash
python PKDP.py --mode predict --geno data/new_samples.csv \
               --model_path results/best_model.pth --output_path predictions/
```


## Version

### v0.0.1
- Initial release of PKDP.
- Added dual-path CNN architecture.
- Integrated support for prior knowledge features.
- Implemented training with cross-validation and hyperparameter optimization.
- Added visualization tools for training progress and predictions.


## Citation
Han F, Gao M, Zhao Y, Bi C, et al. Improving genomic selection accuracy using a dual-path convolutional neural network framework: a terpenoid case study. Unpublished.


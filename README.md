<p align="center"><img src="https://github.com/user-attachments/assets/f5217d49-3385-44fa-9776-8f398829c1fb" style="width: 50%; height: auto;"></p>

<div align="center">
<h1>Prior Knowledge Dual-Path CNN</h1>

</div>

***

<div align="center">

[![Release Version](https://img.shields.io/github/v/release/yunmika/PKDP?color=blue)](https://github.com/yunmika/PKDP/releases)
[![License](https://img.shields.io/github/license/yunmika/PKDP?color=green)](https://github.com/yunmika/PKDP/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/yunmika/PKDP?color=orange)](https://github.com/yunmika/PKDP/commits/main)

</div>

<p  align="center"><img src="https://github.com/user-attachments/assets/089f554e-3743-49a8-9672-61dda37afaa2" style="width: 70%; height: auto;"></p>

<div style="clear: both; margin: 2rem 0; overflow: hidden;">
  <a href="https://github.com/user-attachments/assets/ea8747a8-acbe-49fb-ac59-d912acf12226">
    <img src="https://github.com/user-attachments/assets/ea8747a8-acbe-49fb-ac59-d912acf12226" align="right" style="width: 25%; height: auto; margin-left: 1rem; margin-bottom: 1rem;" alt="Graphical Abstract">
  </a>
  <div style="text-align: justify;">
    <p>PKDP (Prior Knowledge Dual-Path CNN) is a dual-path convolutional neural network framework designed to enhance genomic selection (GS) by integrating genome-wide association study (GWAS) results with genome-wide minor-effect markers.</p>
    <p>Han, F., Gao, M., Zhao, Y., Bi, C., Yang, Y., Zhang, J., Wang, Y. and Chen, Y. (2025), Improving genomic selection accuracy using a dual-path convolutional neural network framework: a terpenoid case study. <em>New Phytol</em>. https://doi.org/10.1111/nph.70727</p>
  </div>
</div>

<div style="clear: both; margin-bottom: 2rem;"></div>

## Installation

```bash
# Create a new conda environment
conda create -n PKDP_env python=3.8

# Activate the environment (works on all platforms)
conda activate PKDP_env

# Install PKDP
git clone https://github.com/yunmika/PKDP.git
cd ./PKDP
chmod +x ./PKDP.py

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
python ./PKDP.py train -h
```

#### Required Parameters
| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `--train_phe`     | Path to the training phenotype file             |
| `--geno`          | Path to the genotype file                       |
| `--output_path`   | Directory to save outputs                       |

#### Optional Parameters
| Parameter                | Description                                      | Default Value |
|--------------------------|--------------------------------------------------|---------------|
| `--pnum`                 | Phenotype column index or name                 | First column  |
| `--prefix`               | Prefix for output files                         | Timestamp     |
| `--batch_size`           | Batch size for training                         | 32            |
| `--epochs`               | Number of training epochs                       | 100           |
| `--optuna_trials`        | Number of Optuna trials for hyperparameter tuning | 50           |
| `--device`               | Device to use (`cuda` or `cpu`)                 | cuda:0        |
| `--optimizer`            | Optimizer type (`Adam`, `SGD`, `AdamW`)         | Adam          |
| `--early_stop`           | Enable early stopping                           | False         |
| `--prior_features`       | Prior knowledge features (space-separated IDs)  | None          |
| `--prior_features_file`  | Path to a file with one prior feature ID per line | None        |
| `--adjust_encoding`      | Adjust genotype encoding from {0,1,2} to {-1,0,1} | False      |


#### Usage

```bash
python ./PKDP.py train \
               --train_phe demo/train_phe.csv \
               --geno demo/train_geno.csv \
               --output_path results/ \
               --prior_features_file ./demo/prior_features.txt
```

#### Notes
In the train model, the training set will automatically calculate hyperparameters through cross-validation. After obtaining the best hyperparameters, the model will be trained using the entire training set. This mode outputs the trained model file, which can be used for prediction in the predict model. For genotypes with values 0/1/2, it is recommended to set `--adjust_encoding`.


### Prediction

```sh
python ./PKDP.py predict -h
```

#### Required Parameters
| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `--geno`          | Path to the genotype file                       |
| `--model_path`    | Path to the trained model file                  |
| `--output_path`   | Directory to save outputs                       |

#### Optional Parameters
| Parameter                | Description                                      | Default Value |
|--------------------------|--------------------------------------------------|---------------|
| `--test_phe`             | Path to the testing phenotype file (optional)   | None          |
| `--pnum`                 | Phenotype column name                 | First column  |
| `--prefix`               | Prefix for output files                         | Timestamp     |
| `--device`               | Device to use (`cuda` or `cpu`)                 | cuda:0        |
| `--adjust_encoding`      | Adjust genotype encoding from {0,1,2} to {-1,0,1} | False      |
| `--prior_features`       | Prior knowledge features (space-separated IDs)  | None          |
| `--prior_features_file`  | Path to a file with one prior feature ID per line | None        |


#### Usage

```bash
python ./PKDP.py predict \
               --geno demo/test_geno.csv \
               --test_phe demo/test_phe.csv \
               --prior_features_file ./demo/prior_features.txt \
               --model_path results/best_model.pth --output_path predictions/
```

#### Notes
- When no prior features are provided, the model will only utilize the main convolutional path.
- It is recommended to use the `--prior_features_file` parameter instead of the `--prior_features` parameter to specify prior features.
- The `--pnum` parameter can be used to specify the phenotype column to predict.
- During model training, samples with NA values in the phenotype will be automatically ignored, so there is no need to manually remove samples with NA values.
- The order of SNPs in the prediction should match the order used during training.
- Input phenotype data format: see `./demo/demo_phenotypes.csv`
- Input genotype data format: see `./demo/demo_genotypes.csv`
- During model training, please pay close attention to adjusting the following hyperparameters, as they significantly impact model performance:
    *   `--main_channels`: Number of channels in the main network.
    *   `--prior_channels`: Number of channels in the prior knowledge network.
    *   `--learning_rate`: Learning rate.
    *   `--batch_size`: Batch size.



## Version

### v0.0.1
- Initial release of PKDP.
- Added dual-path CNN architecture.
- Integrated support for prior knowledge features.
- Implemented training with cross-validation and hyperparameter optimization.
- Added visualization tools for training progress and predictions.
### v0.0.8
- In the previous version, the optional parameter `test_phe` was only used for evaluating the predictive capability after model training. It has been removed in the new version.

## Citation
Han, F., Gao, M., Zhao, Y., Bi, C., Yang, Y., Zhang, J., Wang, Y. and Chen, Y. (2025), Improving genomic selection accuracy using a dual-path convolutional neural network framework: a terpenoid case study. New Phytol. https://doi.org/10.1111/nph.70727


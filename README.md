<p align="center"><img src="https://github.com/user-attachments/assets/f5217d49-3385-44fa-9776-8f398829c1fb" style="width: 50%; height: auto;"></p>

<div align="center">
<h1>Prior Knowledge Dual-Path CNN</h1>

</div>

***

<div align="center">

[![Release Version](https://img.shields.io/badge/release-v0.1.5-blue.svg)](https://github.com/yunmika/PKDP/releases)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/yunmika/PKDP?color=green)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/yunmika/PKDP?color=orange)](https://github.com/yunmika/PKDP/commits/main)

</div>

## Installation

```bash
# Create a new conda environment
conda create -n PKDP_env python=3.8
conda activate PKDP_env

# Install PKDP
git clone https://github.com/yunmika/PKDP.git
cd ./PKDP
chmod +x ./PKDP.py

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training

```bash
python ./PKDP.py train \
    --train_phe demo/demo3_train_phe.csv \
    --geno demo/demo3_train_geno.csv \
    --prior_features_file demo/demo3_prior_features.txt \
    --output_path results_demo3/ \
    --prefix demo3
```

### 2. Prediction

```bash
python ./PKDP.py predict \
    --geno demo/demo3_test_geno.csv \
    --test_phe demo/demo3_test_phe.csv \
    --prior_features_file demo/demo3_prior_features.txt \
    --model_path results_demo3/demo3_best_model.pth \
    --output_path predictions_demo3/ \
    --prefix demo3_pred
```

## Documentation

Detailed documentation and references are available in the [Wiki](wiki/Home.md):

- [Data Preparation](wiki/Data-Preparation.md): Format requirements for genotype, phenotype, and prior feature files.
- [CLI Reference](wiki/CLI-Reference.md): Detailed parameter options for training and prediction.
- [Tutorials and Demos](wiki/Tutorials-and-Demos.md): Benchmark evaluation and usage examples on Demos 1–3.

## Citation

Han, F., Gao, M., Zhao, Y., Bi, C., Yang, Y., Zhang, J., Wang, Y. and Chen, Y. (2025), Improving genomic selection accuracy using a dual-path convolutional neural network framework: a terpenoid case study. *New Phytol*. https://doi.org/10.1111/nph.70727

```bibtex
@article{han2025improving,
  title={Improving genomic selection accuracy using a dual-path convolutional neural network framework: a terpenoid case study},
  author={Han, Fengchen and Gao, Mengfan and Zhao, Yuxuan and Bi, Chen and Yang, Yanzhao and Zhang, Junjie and Wang, Yue and Chen, Yaxian},
  journal={New Phytologist},
  year={2025},
  publisher={Wiley Online Library},
  doi={10.1111/nph.70727}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

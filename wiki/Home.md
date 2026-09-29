# PKDP Documentation

Welcome to the PKDP (Prior Knowledge Dual-Path CNN) documentation wiki.

PKDP is a deep learning framework designed for genomic selection (GS). It integrates a multi-scale convolutional polygenic pathway for continuous linkage disequilibrium (LD) haplotypes with an explicit pairwise epistasis pathway for biological prior QTLs, dynamically combined via Feature-wise Linear Modulation (FiLM).

---

## Table of Contents

- [Installation and Setup](Installation.md)
  - Hardware and OS requirements
  - Conda environment setup
  - GPU / CUDA configuration
- [Data Preparation Guide](Data-Preparation.md)
  - Genotype file format (CSV, {0, 1, 2} and {-1, 0, 1})
  - Phenotype file format
  - Prior knowledge feature list format
- [Command Line Reference](CLI-Reference.md)
  - PKDP.py train: Options and tuning guidelines
  - PKDP.py predict: Model inference and evaluation outputs
- [Tutorials and Demos](Tutorials-and-Demos.md)
  - Demo 1: Additive genetic architecture and LD
  - Demo 2: Additive, dominance, and LD
  - Demo 3: Additive, non-linear epistasis, and LD

---

## Reference Links

- [GitHub Repository](https://github.com/yunmika/PKDP)
- [Published Paper in New Phytologist (2025)](https://doi.org/10.1111/nph.70727)
- [Issue Tracker](https://github.com/yunmika/PKDP/issues)

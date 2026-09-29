# Data Preparation Guide

PKDP requires standard CSV/TXT inputs for training and prediction:
1. **Genotype Matrix** (CSV)
2. **Phenotype Table** (CSV)
3. **Prior Knowledge Features** (TXT or space-separated string)

---

## 1. Genotype File Format (`--geno`)

- **Format**: Comma-separated values (CSV).
- **Index Column**: The first column must be individual sample IDs (e.g., `ID`, `Taxa`, `Line`).
- **Header**: Columns 2 to $P$ are marker IDs (e.g., `SNP_1`, `chr1_102934`).
- **Ordering**: **Markers must be sorted by chromosome and physical base-pair coordinate from left to right.** This ensures the 1D multi-scale convolution captures genuine local linkage disequilibrium (LD) haplotypes.
- **Encoding**:
  - Standard allele dosage: `{0, 1, 2}` representing homozygous reference, heterozygous, and homozygous alternate.
  - If `--adjust_encoding` is set, values are automatically shifted to `{-1, 0, 1}`.

### Example:
```csv
ID,X1_1,X1_2,X1_3,X1_4,X1_5
Line_1,0,0,2,2,0
Line_2,0,0,2,0,0
Line_3,0,0,2,2,0
Line_4,0,0,2,2,0
```

---

## 2. Phenotype File Format (`--train_phe` / `--test_phe`)

- **Format**: Comma-separated values (CSV).
- **Index Column**: The first column must be individual sample IDs matching the genotype file.
- **Header**: Columns contain trait names (e.g., `Yield`, `Height`, `Phenotype`).
- **Missing Values**: Samples with missing phenotype values (`NA`, `NaN`, blank) in the target column are automatically detected and omitted from training.

### Example:
```csv
ID,Phenotype
Line_1,-1.234
Line_2,0.852
Line_3,0.119
Line_4,-0.457
```

---

## 3. Prior Knowledge Features (`--prior_features_file`)

- **Format**: Plain text file (`.txt`), with **one marker ID per line**.
- **Matching**: Marker IDs must match column names in the genotype CSV file.
- **Origin**: Typically obtained from prior GWAS findings, QTL mapping, biological pathways, or functional gene annotations.

### Example (`prior_features.txt`):
```text
X1_14
X1_18
X1_25
X2_41
X3_90
```

> **Note on Feature Selection**:  
> To avoid feature selection data leakage in evaluation, prior markers must be derived from external studies or identified strictly using the training set partition. Do not perform GWAS on the combined training and testing dataset prior to cross-validation.

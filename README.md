# SpecBridge

SpecBridge provides a DreaMS-conditioned adapter for spectra->molecule mapping and a training pipeline with synthetic and real (MGF) data.

## Introduction

### Getting Candidates

To generate candidate molecules for evaluation, use the `build_pubchem_candidates.py` script:

```bash
python build_pubchem_candidates.py input_mgf candidates --max-per 200 --ppm 10 --by-fold test --workers 16
```

This script will:
- Process the MGF file (`input_mgf`)
- Generate candidate molecules from PubChem
- Save candidates to a pickle file (`candidates`)
- Limit to 200 candidates per spectrum (`--max-per 200`)
- Use 10 ppm mass tolerance (`--ppm 10`)
- Filter by fold (e.g., `test`)
- Use 16 worker processes for parallelization

### Getting DreaMS ckpt:
Weights of pre-trained models: [https://zenodo.org/records/10997887](https://zenodo.org/records/10997887)

### Running Evaluation

To evaluate the model on a dataset with candidates:

```bash
python -m specbridge.eval.candidates \
    --mgf input_mgf \
    --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
    --adapter-ckpt runs/specbridge_align_chemberta_pub_v3g/ckpt_000400.pt \
    --candidates candidates \
    --fold-query test \
    --use-mapped \
    --deterministic-map \
    --no-gaussian \
    --batch-size 32 \
    --cond-dim 2048 \
    --mapper-hidden 2048 \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m
```

**Note:** The MGF file will be provided. Make sure to update the paths to your specific MGF file location.

# SpecBridge

SpecBridge provides a DreaMS-conditioned adapter for spectra->molecule mapping and a training pipeline with synthetic and real (MGF) data.

## Introduction

### Getting Candidates

To generate candidate molecules for evaluation, use the `build_pubchem_candidates.py` script:

```bash
python build_pubchem_candidates.py input_mgf candidates --max-per 200 --ppm 10 --by-fold test --workers 16
```

This script will:
- Process the MGF file (`input_mgf`)
- Generate candidate molecules from PubChem
- Save candidates to a pickle file (`candidates`)
- Limit to 200 candidates per spectrum (`--max-per 200`)
- Use 10 ppm mass tolerance (`--ppm 10`)
- Filter by fold (e.g., `test`)
- Use 16 worker processes for parallelization

### Running Evaluation

To evaluate the model on a dataset with candidates:

```bash
python -m specbridge.eval.candidates \
    --mgf input_mgf \
    --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
    --adapter-ckpt runs/specbridge_align_chemberta_pub_v3g/ckpt_000400.pt \
    --candidates candidates \
    --fold-query test \
    --use-mapped \
    --deterministic-map \
    --no-gaussian \
    --batch-size 32 \
    --cond-dim 2048 \
    --mapper-hidden 2048 \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m
```

**Note:** The MGF file will be provided. Make sure to update the paths to your specific MGF file location.



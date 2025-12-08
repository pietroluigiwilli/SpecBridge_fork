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


python dreams_condition_adapter.py   --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf   --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt   --fold train   --batch-size 128 --epochs 2   --cond-dim 2048 --mapper-hidden 2048   --no-gaussian   --supcon-k 4   --w-con 0 --w-con-mapped 0 --w-map 5.0 --w-ortho 1e-3   --w-supcon 1.0 --supcon-temp 0.07   --w-hard 0.0 --hard-topk 16 --hard-temp 0.07   --w-iso 0.5 --iso-k 8 --iso-temp 0.07   --log-every 50 --save-every 200   --outdir runs/specbridge_align_chemberta_pub_v3g_msgym_fintunespec   --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m  --lr 1e-5 --unfreeze-last 2 --unfreeze-after 0 

python dreams_condition_adapter.py   --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MSnLib/combined_ms2_with_folds.mgf   --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt   --fold train   --batch-size 32 --epochs 2   --cond-dim 2048 --mapper-hidden 2048   --no-gaussian   --supcon-k 4   --w-con 0 --w-con-mapped 0.005 --w-map 5.0 --w-ortho 1e-3   --w-supcon 1.0 --supcon-temp 0.07   --w-hard 0.0 --hard-topk 16 --hard-temp 0.07   --w-iso 0.5 --iso-k 8 --iso-temp 0.07   --log-every 50 --save-every 200   --outdir runs/specbridge_align_chemberta_pub_v3g_msblib_contrafintune   --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m  --lr 1e-5 --unfreeze-last 2 --unfreeze-after 0 --unfreeze-mol-last 2 --unfreeze-mol-after 0


python dreams_condition_adapter.py   --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/nist23.mgf   --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt   --fold train   --batch-size 64 --epochs 2   --cond-dim 2048 --mapper-hidden 2048   --no-gaussian   --supcon-k 4   --w-con 0 --w-con-mapped 0.005 --w-map 5.0 --w-ortho 1e-3   --w-supcon 1.0 --supcon-temp 0.07   --w-hard 0.0 --hard-topk 16 --hard-temp 0.07   --w-iso 0.5 --iso-k 8 --iso-temp 0.07   --log-every 50 --save-every 100   --outdir runs/specbridge_align_chemberta_pub_v3g_nist_contrafintune   --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m  --lr 1e-5 --unfreeze-last 2 --unfreeze-after 0 --unfreeze-mol-last 2 --unfreeze-mol-after 0


python dreams_condition_adapter.py   --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf   --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt   --fold train   --batch-size 128 --epochs 2   --cond-dim 2048 --mapper-hidden 2048   --no-gaussian   --supcon-k 4   --w-con 0 --w-con-mapped 0 --w-map 5.0 --w-ortho 1e-3   --w-supcon 1.0 --supcon-temp 0.07   --w-hard 0.0 --hard-topk 16 --hard-temp 0.07   --w-iso 0.5 --iso-k 8 --iso-temp 0.07   --log-every 50 --save-every 200   --outdir runs/specbridge_align_chemberta_pub_v3g_msgym_fintunedrug   --mol-space chemberta --chemberta-model Derify/ChemBERTa-druglike  --lr 1e-5 --unfreeze-last 2 --unfreeze-after 0 --unfreeze-mol-last 2 --unfreeze-mol-after 0


finetune: 

--unfreeze-last 2 --unfreeze-after 0 --unfreeze-mol-last 2 --unfreeze-mol-after 0

CUDA_VISIBLE_DEVICES=1 python -m specbridge.eval.candidates_by_adduct     --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/nist23.mgf     --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt     --adapter-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_nist23_finetune/ckpt_000200.pt    --candidates /cluster/tufts/liulab/yiwan01/SpecBridge/data/cand_dict_merged.pkl --fold-query test --use-mapped --deterministic-map --no-gaussian     --batch-size 32     --cond-dim 2048 --mapper-hidden 2048     --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m --adduct "[M+H]+"

python -m specbridge.eval.predict_smiles \
    --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/all_spectra.mgf \
    --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
    --adapter-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g/ckpt_001400.pt \
    --candidates-json /cluster/tufts/liulab/yiwan01/SpecBridge/data/feature_id_to_candidates.json \
    --output predictions.json \
    --use-mapped --deterministic-map \
    --cond-dim 2048 --mapper-hidden 2048 \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m \
    --handle-duplicates keep_first
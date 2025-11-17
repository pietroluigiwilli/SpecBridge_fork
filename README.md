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

python dreams_condition_adapter.py \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --fold train \
  --batch-size 128 --epochs 10 \
  --cond-dim 2048 --mapper-hidden 2048 \
  --no-gaussian \
  --supcon-k 4 \
  --w-con 0 --w-con-mapped 0 --w-map 5.0 --w-ortho 1e-3 \
  --w-supcon 1.0 --supcon-temp 0.07 \
  --w-hard 0.0 --hard-topk 16 --hard-temp 0.07 \
  --train-candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl \
  --w-iso 0.5 --iso-k 8 --iso-temp 0.07 \
  --log-every 50 --save-every 200 \
  --outdir runs/specbridge_align_chemberta_pub_v3g_ablation_finetune \
  --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m \
  --unfreeze-last 2 --unfreeze-after 0 --unfreeze-mol-last 2 --unfreeze-mol-after 0


python dreams_condition_adapter.py \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --fold train \
  --batch-size 32 --epochs 10 \
  --cond-dim 2048 --mapper-hidden 2048 \
  --no-gaussian \
  --supcon-k 4 \
  --w-con 0 --w-con-mapped 0 --w-map 5.0 --w-ortho 1e-3 \
  --w-supcon 1.0 --supcon-temp 0.07 \
  --w-hard 0.0 --hard-topk 16 --hard-temp 0.07 \
  --train-candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl \
  --w-iso 0.5 --iso-k 8 --iso-temp 0.07 \
  --log-every 50 --save-every 200 \
  --outdir runs/specbridge_align_chemberta_pub_v3g_ablation_scratch \
  --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m \
  --init-spec-from-scratch --init-mol-from-scratch

python dreams_condition_adapter.py \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --fold train \
  --batch-size 128 --epochs 10 \
  --cond-dim 2048 --mapper-hidden 2048 \
  --no-gaussian \
  --supcon-k 4 \
  --w-con 0 --w-con-mapped 0 --w-map 5.0 --w-ortho 1e-3 \
  --w-supcon 1.0 --supcon-temp 0.07 \
  --w-hard 0.0 --hard-topk 16 --hard-temp 0.07 \
  --w-iso 0.5 --iso-k 8 --iso-temp 0.07 \
  --log-every 50 --save-every 200 \
  --outdir runs/specbridge_align_chemberta_pub_v3g_spectraverse \
  --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m


python dreams_condition_adapter.py   --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/canopus_10k.mgf   --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt   --fold train   --batch-size 16 --epochs 1000   --cond-dim 2048 --mapper-hidden 2048   --no-gaussian   --supcon-k 4   --w-con 0 --w-con-mapped 0 --w-map 5.0 --w-ortho 1e-3   --w-supcon 1.0 --supcon-temp 0.07 --lr 1e-4  --w-hard 0.0 --hard-topk 16 --hard-temp 0.07   --train-candidates /cluster/tufts/liulab/yiwan01/SpecBridge/data/cand_dict_large_smiles.pkl   --w-iso 0.5 --iso-k 8 --iso-temp 0.07   --log-every 100 --save-every 1000   --outdir runs/specbridge_align_chemberta_pub_v3g_canopus   --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m --resume /cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_freeze/ckpt_001400.pt 


  python -m specbridge.eval.candidates \
    --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
    --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
    --adapter-ckpt runs/specbridge_align_chemberta_v4/ckpt_007000.pt \
    --candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl \
    --fold-query test --use-mapped --deterministic-map \
    --batch-size 32 \
    --cond-dim 2048 --mapper-hidden 2048 \
    --mol-space chemberta --chemberta-model seyonec/ChemBERTa-zinc-base-v1 \
    --cache-cand-emb /cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_test_chemberta_all.pt




python -m specbridge.eval.candidates     --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf     --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt     --adapter-ckpt runs/specbridge_align_chemberta_v3b/ckpt_007000.pt     --candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl --fold-query test --use-mapped --deterministic-map     --batch-size 32     --cond-dim 2048 --mapper-hidden 2048     --mol-space chemberta --chemberta-model seyonec/ChemBERTa-zinc-base-v1 --no-gaussian



python -m specbridge.eval.candidates     --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf     --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt     --adapter-ckpt runs/specbridge_align_chemberta_pub_v3f/ckpt_000600.pt    --candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl --fold-query test --use-mapped --deterministic-map --no-gaussian     --batch-size 32     --cond-dim 2048 --mapper-hidden 2048     --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m



python -m specbridge.eval.generator \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --adapter-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3f/ckpt_001400.pt \
  --cond-dim 2048 --mapper-hidden 2048 --no-gaussian \
  --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m \
  --molgpt-ckpt /cluster/tufts/liulab/yiwan01/molgpt/cond_gpt/weights/pub_enhence.pt \
  --molgpt-config /cluster/tufts/liulab/yiwan01/molgpt/cond_gpt/weights/molgpt_infer_config_pub13m.json \
  --vocab-json /cluster/tufts/liulab/yiwan01/molgpt/specbridge_stoi.json \
  --fold-query train \
  --samples-per-spec 64 --max-new-tokens 512 --temperature 0.8 --top-k 10 \
  --out-csv runs/molgpt_gen_from_spec.csv


# SpecBridge

SpecBridge provides a DreaMS-conditioned adapter for spectra->molecule mapping and a training pipeline with synthetic and real (MGF) data.

## Install (editable)

```bash
pip install -e .
```

## CLI

```bash
specbridge --help
```

- Synthetic demo:
```bash
specbridge --demo --steps 200
```
- Real training:
```bash
specbridge \
  --mgf /path/to/data.mgf \
  --dreams-ckpt /path/to/ssl_model.ckpt \
  --batch-size 64 --epochs 1 --log-every 50
```

## Legacy script (still runnable)

```bash
python specbridge_entry.py --demo --steps 200
```

python dreams_condition_adapter.py \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --fold train \
  --batch-size 128 --epochs 100 \
  --cond-dim 2048 --mapper-hidden 2048 \
  --no-gaussian \
  --supcon-k 4 \
  --w-con 0 --w-con-mapped 1.0 --w-map 5.0 --w-ortho 1e-3 \
  --w-supcon 1.0 --supcon-temp 0.07 \
  --w-hard 0.0 --hard-topk 16 --hard-temp 0.07 \
  --train-candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl \
  --w-iso 0.5 --iso-k 8 --iso-temp 0.07 \
  --log-every 50 --save-every 200 \
  --outdir runs/specbridge_align_chemberta_v4 \
  --mol-space chemberta --chemberta-model seyonec/ChemBERTa-zinc-base-v1 




  python -m specbridge.eval.candidates \
    --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
    --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
    --adapter-ckpt runs/specbridge_align_chemberta_v3/ckpt_007000.pt \
    --candidates /cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl \
    --fold-query test --use-mapped --deterministic-map \
    --batch-size 32 --limit 1000 \
    --cond-dim 2048 --mapper-hidden 2048 \
    --mol-space chemberta --chemberta-model seyonec/ChemBERTa-zinc-base-v1 \
    --cache-cand-emb /cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_test_chemberta.pt


    python dreams_condition_adapter.py \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --fold train \
  --batch-size 128 --epochs 8 \
  --cond-dim 2048 --mapper-hidden 2048 \
  --no-gaussian \
  --supcon-k 4 \
  --w-con 0.2 --w-con-mapped 0.2 --w-map 1.0 --w-ortho 1e-3 \
  --w-supcon 1.0 --supcon-temp 0.07 \
  --w-hard 1.0 --hard-topk 8 --hard-temp 0.07 \
  --w-dec 0.0 --w-fwd 0.0 \
  --log-every 50 --save-every 200 \
  --outdir runs/specbridge_align_chemberta_v4 \
  --mol-space chemberta --chemberta-model seyonec/ChemBERTa-zinc-base-v1 \
  --lr 2e-4 --lr-scheduler cosine --lr-warmup-ratio 0.05 --lr-min-mult 0.1


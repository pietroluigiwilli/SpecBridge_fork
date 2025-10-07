#!/bin/bash
#SBATCH -n 48            # cpu cores
#SBATCH -t 7-00:00          # Runtime in D-HH:MM
#SBATCH -p gpu
##SBATCH --mem-per-cpu=8000  # Memory (in MB) per cpu
#SBATCH --mem=512g
#SBATCH -o ./log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e ./log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --gres=gpu:a100
#SBATCH -x s1cmp003
##SBATCH -w s1cmp004,s1cmp001,s1cmp003,s1cmp005
source activate specbridge
python dreams_condition_adapter.py \
  --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf \
  --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
  --fold train \
  --batch-size 128 --epochs 50 \
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
  --mol-space chemberta \
  --chemberta-model seyonec/ChemBERTa-zinc-base-v1 \
  --val-every 200  \
  --early-stop \
  --patience 50

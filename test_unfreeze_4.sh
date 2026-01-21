#!/bin/bash
# Quick test to run unfreeze_4 manually

MGF_FILE="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
DREAMS_CKPT="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"
BASE_OUTDIR="runs/optional_ablations_spectraverse"

BATCH_SIZE=16
EPOCHS=2
LR=1e-4
COND_DIM=2048
LOG_EVERY=50
SAVE_EVERY=200
W_MAP=5.0
W_CON_MAPPED=0
W_ORTHO=1e-3

config_name="unfreeze_4"
outdir="${BASE_OUTDIR}/${config_name}"
log_file="${BASE_OUTDIR}/${config_name}.log"

mkdir -p "$outdir"

cmd="CUDA_VISIBLE_DEVICES=0 python dreams_condition_adapter.py"
cmd="$cmd --mgf $MGF_FILE"
cmd="$cmd --dreams-ckpt $DREAMS_CKPT"
cmd="$cmd --fold train"
cmd="$cmd --batch-size $BATCH_SIZE"
cmd="$cmd --epochs $EPOCHS"
cmd="$cmd --cond-dim $COND_DIM"
cmd="$cmd --no-gaussian"
cmd="$cmd --w-map $W_MAP"
cmd="$cmd --w-con-mapped $W_CON_MAPPED"
cmd="$cmd --w-ortho $W_ORTHO"
cmd="$cmd --log-every $LOG_EVERY"
cmd="$cmd --save-every $SAVE_EVERY"
cmd="$cmd --outdir $outdir"
cmd="$cmd --mol-space chemberta"
cmd="$cmd --chemberta-model $CHEMBERTA_MODEL"
cmd="$cmd --lr $LR"
cmd="$cmd --n-blocks 8 --mapper-hidden 2048 --unfreeze-last 4 --unfreeze-after 0"

echo "Command: $cmd"
echo "Output dir: $outdir"
echo "Log file: $log_file"
echo ""
echo "To run: $cmd > $log_file 2>&1 &"

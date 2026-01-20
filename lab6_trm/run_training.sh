#!/bin/bash
#SBATCH --job-name=train_trm
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=plgar2025-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=train.log
#SBATCH --error=train.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

cd /net/tscratch/people/plgboksa/lab6_comp_ling_trm

source .venv/bin/activate

uv run train.py \
    --data-dir data \
    --epochs 3 \
    --batch-size 512 \
    --lr 1e-4 \
    --wandb \
    --wandb-project trm-sudoku \
    --wandb-run-name lion_lr_1e-4_bigger \
    --H-cycles 3 \
    --L-cycles 6 \
    --hidden-size 512 \
    --eval-every 500 \
    --save-every 2000 \
    --eval-samples 25000 \
    --num-workers 8
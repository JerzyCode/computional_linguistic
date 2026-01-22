#!/bin/bash
#SBATCH --job-name=train_trm
#SBATCH --time=14:00:00
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
    --data-dir large-data \
    --epochs 1 \
    --batch-size 768 \
    --lr 1.5e-4 \
    --weight-decay 1.0 \
    --wandb \
    --wandb-project trm-sudoku \
    --wandb-run-name lion_lr_1.5e-4_run1 \
    --H-cycles 3 \
    --L-cycles 6 \
    --max-steps 8 \
    --hidden-size 384 \
    --eval-every 100 \
    --save-every 500 \
    --eval-samples 2912 \
    --num-workers 8


    # uv run train.py --data-dir data --epochs 1 --batch-size 512 --lr 1e-4 --wandb --wandb-project trm-sudoku-find-lr --wandb-run-name lion_lr_1e-4_bigger --H-cycles 3 --L-cycles 6 --hidden-size 512 --eval-every 100 --save-every 1000


    # Full dataset: 3831994 train, 422786 test

#!/usr/bin/zsh

echo "Current machine: $(hostname)"

srun --partition=c23g --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --time=00:35:00 --job-name=cnn --output=logs/stdout_%j.txt ./run.sh

#!/bin/bash
# Run all three persona vector experiments sequentially

# Activate virtual environment
source venv/bin/activate

# Run experiments sequentially
echo "Starting experiment 1: qwen2.5_7b_ape_inc_coef3"
python main.py experiment=qwen2.5_7b_ape_inc_coef3

echo "Starting experiment 2: qwen2.5_7b_ape_inc_coef5"
python main.py experiment=qwen2.5_7b_ape_inc_coef5

echo "Starting experiment 3: qwen2.5_7b_ape_L15_20_25_c2p0"
python main.py experiment=qwen2.5_7b_ape_L15_20_25_c2p0

echo "All experiments completed!"


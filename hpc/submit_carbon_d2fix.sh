#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=pallas_c_d2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4

# D2-fix carbon re-benchmark. Submit with: sbatch --export=ALL,SEED=42 ...
source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate nequip
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/lz432/pallas_d2fix/Pallas2:/scratch/lz432/pallas_d2fix/torch-fplib:$PYTHONPATH
python -u run_carbon_15gpa.py "${SEED:-42}" bench/carbon-d2fix-20260731

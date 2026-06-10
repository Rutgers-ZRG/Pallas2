#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=pallas_c_nodrag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
eval "$(conda shell.bash hook)" && conda activate nequip
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/lz432/pallas_nodrag/Pallas2:/scratch/lz432/pallas_nodrag/torch-fplib:$PYTHONPATH
python -u run_carbon_nodrag.py

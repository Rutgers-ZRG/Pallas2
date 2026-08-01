#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=pallas_c
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4

# Carbon search. Submit FROM a wd_sNN dir: sbatch --export=ALL,SEED=42,TAG=bench/... submit_carbon_tag.sh
source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate nequip
cd "$SLURM_SUBMIT_DIR"
DEST=$(dirname "$SLURM_SUBMIT_DIR")
export PYTHONPATH=$DEST/Pallas2:$DEST/torch-fplib:$PYTHONPATH
python -u run_carbon_15gpa.py "${SEED:-42}" "${TAG:-untagged}"

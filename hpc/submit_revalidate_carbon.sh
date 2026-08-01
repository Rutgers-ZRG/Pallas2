#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=pallas_reval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4

# Path-only revalidate of one carbon workdir. Submit FROM the staging DEST:
#   sbatch --export=ALL,WD=wd_s42 submit_revalidate_carbon.sh
source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate nequip
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$SLURM_SUBMIT_DIR/Pallas2:$SLURM_SUBMIT_DIR/torch-fplib:$PYTHONPATH
export ALLEGRO_MODEL=/scratch/lz432/allegro_r2scan_finetune/allegro_r2scan_carbon.nequip.pth
export PALLAS_DEVICE=cuda
python -u Pallas2/benchmarks/revalidate.py "${WD:?}" --calc allegro-carbon --path-only

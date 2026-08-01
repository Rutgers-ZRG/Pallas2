#!/bin/bash
# Stage PALLAS at TAG + torch-fplib + carbon driver/submit scripts to a fresh
# Amarel scratch DEST (prior benchmark dirs untouched).
# Usage: stage_amarel_tag.sh TAG DEST   e.g.
#   stage_amarel_tag.sh bench/carbon-d1fix-20260801 /scratch/lz432/pallas_d1fix
set -euo pipefail
TAG=${1:?usage: stage_amarel_tag.sh TAG DEST}
DEST=${2:?usage: stage_amarel_tag.sh TAG DEST}

cd /Users/li/dev/Pallas2
git archive --prefix=Pallas2/ -o /tmp/pallas2_stage.tar.gz "$TAG"
tar -C /Users/li/dev -czf /tmp/torchfplib_stage.tar.gz \
    --exclude='*/.git' --exclude='*/__pycache__' torch-fplib

ssh an "mkdir -p $DEST/wd_s42 $DEST/wd_s43 $DEST/wd_s44"
scp /tmp/pallas2_stage.tar.gz /tmp/torchfplib_stage.tar.gz an:$DEST/
ssh an "cd $DEST && tar xzf pallas2_stage.tar.gz && tar xzf torchfplib_stage.tar.gz"
for s in 42 43 44; do
  scp benchmarks/carbon/run_carbon_15gpa.py hpc/submit_carbon_tag.sh an:$DEST/wd_s$s/
done
scp hpc/submit_revalidate_carbon.sh an:$DEST/
echo "Staged $TAG -> an:$DEST (wd_s42/43/44)"

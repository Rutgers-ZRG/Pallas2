#!/bin/bash
# Stage tagged PALLAS code + torch-fplib + carbon driver to Amarel scratch.
# Never touches /scratch/lz432/pallas_carbon_test/ (prior drag-run data).
set -euo pipefail

TAG=bench/carbon-nodrag-20260610
DEST=/scratch/lz432/pallas_nodrag

cd /Users/li/dev/Pallas2
git archive --prefix=Pallas2/ -o /tmp/pallas2_stage.tar.gz "$TAG"
tar -C /Users/li/dev -czf /tmp/torchfplib_stage.tar.gz \
    --exclude='*/.git' --exclude='*/__pycache__' torch-fplib

ssh an "mkdir -p $DEST/workdir_carbon"
scp /tmp/pallas2_stage.tar.gz /tmp/torchfplib_stage.tar.gz an:$DEST/
ssh an "cd $DEST && tar xzf pallas2_stage.tar.gz && tar xzf torchfplib_stage.tar.gz"
scp benchmarks/carbon/run_carbon_nodrag.py hpc/submit_carbon.sh an:$DEST/workdir_carbon/
echo "Staged to an:$DEST/workdir_carbon"

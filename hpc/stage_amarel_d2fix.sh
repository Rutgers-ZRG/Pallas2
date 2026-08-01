#!/bin/bash
# Stage D2-fix PALLAS code + torch-fplib + carbon driver to a FRESH Amarel
# scratch dir (leaves /scratch/lz432/pallas_15gpa/ v1.0.1 artifacts intact).
set -euo pipefail

TAG=bench/carbon-d2fix-20260731
DEST=/scratch/lz432/pallas_d2fix

cd /Users/li/dev/Pallas2
git archive --prefix=Pallas2/ -o /tmp/pallas2_d2.tar.gz "$TAG"
tar -C /Users/li/dev -czf /tmp/torchfplib_d2.tar.gz \
    --exclude='*/.git' --exclude='*/__pycache__' torch-fplib

ssh an "mkdir -p $DEST/wd_s42 $DEST/wd_s43 $DEST/wd_s44"
scp /tmp/pallas2_d2.tar.gz /tmp/torchfplib_d2.tar.gz an:$DEST/
ssh an "cd $DEST && tar xzf pallas2_d2.tar.gz && tar xzf torchfplib_d2.tar.gz"
for s in 42 43 44; do
  scp benchmarks/carbon/run_carbon_15gpa.py hpc/submit_carbon_d2fix.sh \
      an:$DEST/wd_s$s/
done
echo "Staged to an:$DEST (wd_s42/43/44)"

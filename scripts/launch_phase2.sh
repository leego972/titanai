#!/bin/bash
cd /workspace/titanai
mkdir -p checkpoints/titan_v2_pretrain logs
exec python3 -u scripts/pretrain_titan_v2.py \
  --init-from checkpoints/titan_v2_init/initial.pt \
  --out-dir checkpoints/titan_v2_pretrain \
  --max-steps 100000 \
  --eval-every 500 --save-every 1000 --log-every 25 --warmup-steps 500 \
  --batch-size 2 --grad-accum 16 \
  --lr 1e-4 --lr-min-ratio 0.1

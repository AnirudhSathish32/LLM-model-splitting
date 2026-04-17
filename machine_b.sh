#!/bin/bash
# Run this on Machine B (the worker — needs the model downloaded too,
# or point MODEL_PATH at a network share / copied model folder)

MASTER_IP="192.168.100.1"   # Machine A's IP — same on both scripts
MASTER_PORT="29500"

echo "=== Machine B (Worker) ==="
echo "Connecting to master at $MASTER_IP:$MASTER_PORT"
echo ""

torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=$MASTER_IP \
  --master_port=$MASTER_PORT \
  inference.py
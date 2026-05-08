#!/bin/bash
# Run this on Machine A (the one with the model downloaded)
# Make sure Machine B is already running launch_machine_b.sh before running this
export USE_LIBUV=0
export GLOO_SOCKET_INFAME=eth0


MASTER_IP="192.168.4.37"   # Machine A's static IP (set during ethernet setup)
MASTER_PORT="29500"
MODEL_PATH="./llama-3b"

echo "=== Machine A (Master) ==="
echo "Master IP: $MASTER_IP"
echo "Waiting for Machine B to be ready..."
echo ""

torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=$MASTER_IP \
  --master_port=$MASTER_PORT \
  inference.py
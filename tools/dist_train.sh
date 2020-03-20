#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=2 $(dirname "$0")/train.py ../configs/car_cfg.py --launcher pytorch ${@:3}

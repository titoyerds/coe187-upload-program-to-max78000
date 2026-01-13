#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir sdk/Examples/MAX78000/CNN --prefix ripe-unripe --checkpoint-file trained/best-quantized.pth.tar --config-file networks/ripe-unripe-hwc.yaml --fifo --softmax --device MAX78000 --timer 0 --display-checkpoint --verbose
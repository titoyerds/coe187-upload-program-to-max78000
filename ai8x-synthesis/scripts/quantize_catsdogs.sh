#!/bin/sh
# python quantize.py trained/ai85-catsdogs-qat8.pth.tar trained/ai85-catsdogs-qat8-q.pth.tar --device MAX78000 -v
python quantize.py trained/qat_best.pth.tar trained/qat_best-q.pth.tar --device MAX78000 -v

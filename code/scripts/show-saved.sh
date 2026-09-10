#!/bin/bash

SCENE=160906_ian1

python visualize.py \
    stats/eval/lkalman/$SCENE.json \
    --gt-path stats/gt/$SCENE.json


#!/bin/bash

SCENE=160906_ian1
if [ -n "$1" ]; then
    SCENE=$1
fi

KIND=lkalman
if [ -n "$2" ]; then
    KIND=$2
fi

python visualize.py \
    stats/eval/$KIND/$SCENE.json \
    --gt-path stats/gt/$SCENE.json


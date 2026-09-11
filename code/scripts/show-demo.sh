#!/bin/bash

SCENE=pizza
if [ -n "$1" ]; then
    SCENE=$1
fi

python visualize.py \
    stats/demo/$SCENE.json \
    --gt-path stats/gt/$SCENE.json \
    --env-path stats/gt/full.json


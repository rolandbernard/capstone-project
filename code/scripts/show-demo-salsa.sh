#!/bin/bash

SCENE=salsa
if [ -n "$1" ]; then
    SCENE=$1
    shift
fi

python visualize.py \
    stats/demo/$SCENE.json \
    --no-cloud --no-gt --gt-path stats/demo/salsa.json $*


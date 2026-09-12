#!/bin/bash

SCENE=160906_ian1
if [ -n "$1" ]; then
    SCENE=$1
    shift
fi

KIND=tuned
if [ -n "$1" ]; then
    KIND=$1
    shift
fi

python visualize.py \
    stats/eval/$KIND/$SCENE.json \
    --gt-path stats/gt/$SCENE.json \
    --env-path stats/gt/full.json $*


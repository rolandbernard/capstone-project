#!/bin/bash

SCENE=160224_haggling1
if [ -n "$1" ]; then
    SCENE=$1
    shift
fi

python demo.py \
    data/panoptic/$SCENE/vga_01_01.mp4 data/panoptic/$SCENE/vga_19_14.mp4  \
    data/panoptic/$SCENE/vga_14_03.mp4 data/panoptic/$SCENE/vga_06_15.mp4  \
    --cams data/cams/cam01_01.json data/cams/cam19_14.json \
           data/cams/cam14_03.json data/cams/cam06_15.json \
    --scale 0.01 --learned-yolo $*
#     --use-learned --learned-yolo


#!/bin/bash

set -e

echo "Downloading YOLOv4 config and weights..."
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -q --show-progress --no-clobber
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -q --show-progress --no-clobber

echo
echo "Creating YOLOv4-288, YOLOv4-416 and YOLOv4-608 configs..."
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=288/' | sed -e '8s/height=608/height=288/' > yolov4-288.cfg
ln -sf yolov4.weights yolov4-288.weights
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=416/' | sed -e '8s/height=608/height=416/' > yolov4-416.cfg
ln -sf yolov4.weights yolov4-416.weights
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' > yolov4-608.cfg
ln -sf yolov4.weights yolov4-608.weights

echo
echo "Downloading YOLOv4-tiny config and weights..."
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -q --show-progress --no-clobber
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -q --show-progress --no-clobber

echo
echo "Creating YOLOv4-Tiny-288 and YOLOv4-Tiny-416 configs..."
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov4-tiny-288.cfg
echo >> yolov4-tiny-288.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-288.weights
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' > yolov4-tiny-416.cfg
echo >> yolov4-tiny-416.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-416.weights

echo
echo "Done."

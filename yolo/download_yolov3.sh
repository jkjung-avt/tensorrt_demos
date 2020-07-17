#!/bin/bash

set -e

echo "Downloading YOLOv3 config and weights..."
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -q --show-progress --no-clobber
wget https://pjreddie.com/media/files/yolov3.weights -q --show-progress --no-clobber

echo
echo "Creating YOLOv3-288, YOLOv3-416 and YOLOv3-608 configs..."
cat yolov3.cfg | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov3-288.cfg
ln -sf yolov3.weights yolov3-288.weights
cat yolov3.cfg | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov3-416.cfg
ln -sf yolov3.weights yolov3-416.weights
cp yolov3.cfg yolov3-608.cfg
ln -sf yolov3.weights yolov3-608.weights

echo
echo "Downloading YOLOv3-tiny config and weights..."
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -q --show-progress --no-clobber
wget https://pjreddie.com/media/files/yolov3-tiny.weights -q --show-progress --no-clobber

echo
echo "Creating YOLOv3-Tiny-288 and YOLOv3-Tiny-416 configs..."
cat yolov3-tiny.cfg | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov3-tiny-288.cfg
echo >> yolov3-tiny-288.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-288.weights
cp yolov3-tiny.cfg yolov3-tiny-416.cfg
echo >> yolov3-tiny-416.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-416.weights

echo
echo "Done."


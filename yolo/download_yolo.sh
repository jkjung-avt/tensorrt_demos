#!/bin/bash

set -e

# yolov3-tiny
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg -q --show-progress --no-clobber
wget https://pjreddie.com/media/files/yolov3-tiny.weights -q --show-progress --no-clobber

# yolov3
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -q --show-progress --no-clobber
wget https://pjreddie.com/media/files/yolov3.weights -q --show-progress --no-clobber

# yolov3-spp
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-spp.cfg -q --show-progress --no-clobber
wget https://pjreddie.com/media/files/yolov3-spp.weights -q --show-progress --no-clobber

# yolov4-tiny
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -q --show-progress --no-clobber
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -q --show-progress --no-clobber

# yolov4
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -q --show-progress --no-clobber
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -q --show-progress --no-clobber

echo
echo "Creating yolov3-tiny-288.cfg and yolov3-tiny-288.weights"
cat yolov3-tiny.cfg | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov3-tiny-288.cfg
echo >> yolov3-tiny-288.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-288.weights
echo "Creating yolov3-tiny-416.cfg and yolov3-tiny-416.weights"
cp yolov3-tiny.cfg yolov3-tiny-416.cfg
echo >> yolov3-tiny-416.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-416.weights

echo "Creating yolov3-288.cfg and yolov3-288.weights"
cat yolov3.cfg | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov3-288.cfg
ln -sf yolov3.weights yolov3-288.weights
echo "Creating yolov3-416.cfg and yolov3-416.weights"
cat yolov3.cfg | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov3-416.cfg
ln -sf yolov3.weights yolov3-416.weights
echo "Creating yolov3-608.cfg and yolov3-608.weights"
cp yolov3.cfg yolov3-608.cfg
ln -sf yolov3.weights yolov3-608.weights

echo "Creating yolov3-spp-288.cfg and yolov3-spp-288.weights"
cat yolov3-spp.cfg | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov3-spp-288.cfg
ln -sf yolov3-spp.weights yolov3-spp-288.weights
echo "Creating yolov3-spp-416.cfg and yolov3-spp-416.weights"
cat yolov3-spp.cfg | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov3-spp-416.cfg
ln -sf yolov3-spp.weights yolov3-spp-416.weights
echo "Creating yolov3-spp-608.cfg and yolov3-spp-608.weights"
cp yolov3-spp.cfg yolov3-spp-608.cfg
ln -sf yolov3-spp.weights yolov3-spp-608.weights

echo "Creating yolov4-tiny-288.cfg and yolov4-tiny-288.weights"
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov4-tiny-288.cfg
echo >> yolov4-tiny-288.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-288.weights
echo "Creating yolov4-tiny-416.cfg and yolov4-tiny-416.weights"
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' > yolov4-tiny-416.cfg
echo >> yolov4-tiny-416.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-416.weights

echo "Creating yolov4-288.cfg and yolov4-288.weights"
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=288/' | sed -e '8s/height=608/height=288/' > yolov4-288.cfg
ln -sf yolov4.weights yolov4-288.weights
echo "Creating yolov4-416.cfg and yolov4-416.weights"
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=416/' | sed -e '8s/height=608/height=416/' > yolov4-416.cfg
ln -sf yolov4.weights yolov4-416.weights
echo "Creating yolov4-608.cfg and yolov4-608.weights"
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' > yolov4-608.cfg
ln -sf yolov4.weights yolov4-608.weights

echo
echo "Done."


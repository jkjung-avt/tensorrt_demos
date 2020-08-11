#!/bin/bash

set -e

for model in yolov3-288 yolov3-416 yolov3-608 yolov3-tiny-288 yolov3-tiny-416 \
             yolov4-288 yolov4-416 yolov4-608 yolov4-tiny-288 yolov4-tiny-416;
do
  if [[ -f ${model}.onnx ]]; then
    model_left=${model%-*}
    model_right=${model##*-}
    model_opt=${model_left}-opt-${model_right}
    echo Creating symbolic links for ${model_opt}
    ln -sf ${model}.cfg ${model_opt}.cfg
    # weights file is not needed
    ln -sf ${model}.onnx ${model_opt}.onnx
  fi
done

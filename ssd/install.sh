#!/bin/bash

set -e

# install pycuda if necessary
if ! python3 -c "import pycuda" > /dev/null 2>&1; then
  ./install_pycuda.sh
fi

echo "** Patch 'graphsurgeon.py' in TensorRT"

script_path=$(realpath $0)
gs_path=$(ls /usr/lib/python3.?/dist-packages/graphsurgeon/node_manipulation.py)
patch_path=$(dirname $script_path)/graphsurgeon.patch

if head -30 ${gs_path} | tail -1 | grep -q NodeDef; then
  # This is for JetPack-4.2
  sudo patch -N -p1 -r - ${gs_path} ${patch_path}-4.2 && echo
fi
if head -22 ${gs_path} | tail -1 | grep -q update_node; then
  # This is for JetPack-4.2.2
  sudo patch -N -p1 -r - ${gs_path} ${patch_path}-4.2.2 && echo
fi
if head -69 ${gs_path} | tail -1 | grep -q update_node; then
  # This is for JetPack-4.4
  sudo patch -N -p1 -r - ${gs_path} ${patch_path}-4.4 && echo
fi

echo "** Making symbolic link of libflattenconcat.so"

trt_version=$(echo /usr/lib/aarch64-linux-gnu/libnvinfer.so.? | cut -d '.' -f 3)
if [ "${trt_version}" = "5" ] || [ "${trt_version}" = "6" ]; then
  ln -sf libflattenconcat.so.${trt_version} libflattenconcat.so
fi

echo "** Installation done"

#!/bin/bash
#
# Reference for installing 'pycuda': https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu

set -e

folder=${HOME}/src
mkdir -p $folder
script_path=$(realpath $0)
boost_pylib=$(basename /usr/lib/aarch64-linux-gnu/libboost_python3-py3?.so)
boost_pylibname=${boost_pylib%.so}
boost_pyname=${boost_pylibname/lib/}
gs_path=$(ls /usr/lib/python3.?/dist-packages/graphsurgeon/node_manipulation.py)
patch_path=$(dirname $script_path)/graphsurgeon.patch

echo "** Install requirements"
sudo apt-get install -y build-essential python-dev
sudo apt-get install -y libboost-python-dev libboost-thread-dev
sudo pip3 install setuptools

echo "** Download pycuda-2019.1.2 sources"
pushd $folder
if [ ! -f pycuda-2019.1.2.tar.gz ]; then
  wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
fi

echo "** Build and install pycuda-2019.1.2"
tar xzvf pycuda-2019.1.2.tar.gz
cd pycuda-2019.1.2
./configure.py --python-exe=/usr/bin/python3 --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib/aarch64-linux-gnu --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib/aarch64-linux-gnu --boost-python-libname=${boost_pyname} --boost-thread-libname=boost_thread --no-use-shipped-boost
make -j4
python3 setup.py build
sudo python3 setup.py install
python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"

echo "** Patch 'graphsurgeon.py' in TensorRT"
if head -30 ${gs_path} | tail -1 | grep -q NodeDef; then
  # This is for JetPack-4.2
  sudo patch -N -p1 -r - ${gs_path} ${patch_path}-4.2 && echo
fi
if head -22 ${gs_path} | tail -1 | grep -q update_node; then
  # This is for JetPack-4.2.2
  sudo patch -N -p1 -r - ${gs_path} ${patch_path}-4.2.2 && echo
fi

echo "** Installation done"

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
# ported from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
'''

import ctypes

CUDA_SUCCESS = 0

def get_gpu_archs():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return

    gpu_archs = set()

    n_gpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print('cuInit failed with error code %d: %s' % (result, error_str.value.decode()))
        return []

    result = cuda.cuDeviceGetCount(ctypes.byref(n_gpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        # print('cuDeviceGetCount failed with error code %d: %s' % (result, error_str.value.decode()))
        return []

    for i in range(n_gpus.value):
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            gpu_archs.add(str(cc_major.value) + str(cc_minor.value))

    return list(gpu_archs)

if __name__ == '__main__':
    print(' '.join(get_gpu_archs()))

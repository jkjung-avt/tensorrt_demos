from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

library_dirs = [
    '/usr/local/cuda/lib64',
    '/usr/local/TensorRT-7.1.3.4/lib',  # for my x86_64 PC
    '/usr/local/lib',
]

libraries = [
    'nvinfer',
    'cudnn',
    'cublas',
    'cudart_static',
    'nvToolsExt',
    'cudart',
    'rt',
]

include_dirs = [
    # in case the following numpy include path does not work, you
    # could replace it manually with, say,
    # '-I/usr/local/lib/python3.6/dist-packages/numpy/core/include',
    '-I' + numpy.__path__[0] + '/core/include',
    '-I/usr/local/cuda/include',
    '-I/usr/local/TensorRT-7.1.3.4/include',  # for my x86_64 PC
    '-I/usr/local/include',
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(
        Extension(
            'pytrt',
            sources=['pytrt.pyx'],
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=['-O3', '-std=c++11'] + include_dirs
        ),
        compiler_directives={'language_level': '3'}
    )
)

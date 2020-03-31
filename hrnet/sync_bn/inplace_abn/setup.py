from os import path

_src_path = path.join(path.dirname(path.abspath(__file__)), "src")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inplace_abn_cpp_backend',
    ext_modules=[
        CUDAExtension(
            name='inplace_abn_cpp_backend',
            sources=[
              "src/inplace_abn.cpp",
              "src/inplace_abn_cpu.cpp",
              "src/inplace_abn_cuda.cu"
            ],
            extra_compile_args = {
                "cxx":["-O3"],
                'nvcc': ['--expt-extended-lambda']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='double_ext',
      ext_modules=[cpp_extension.CUDAExtension('double_ext', 
        ['double_ext.cpp'],
        libraries=['double_kernel'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
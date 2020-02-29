from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pytorch_aegis',
      ext_modules=[cpp_extension.CUDAExtension('pytorch_aegis', 
        ['aegis.cpp'],
        libraries=['AegisEngine'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
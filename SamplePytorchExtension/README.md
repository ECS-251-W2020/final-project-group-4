# C++/CUDA Extensions in PyTorch

An example of writing a C++ and CUDA extension for PyTorch. See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

# Dependencies

- Visual Studio 2017 community
- Cuda 10.1
- Python 3.7.2
- Pytorch 1.4.0 stable

# Guide

- Give the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,
- Build C++ extensions by going into the `cpp/` folder and executing `python setup.py install`,
- Build CUDA extensions by going into the `cuda/` folder and executing `python setup.py install`, but it is still unfinished,
<!-- - JIT-compile C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and calling `python jit.py`, which will JIT-compile the extension and load it, -->
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {py, cpp, cuda} [--cuda]`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp, cuda} [--cuda]`.
- Run output checks on the code by running `python check.py {forward, backward} [--cuda]`.


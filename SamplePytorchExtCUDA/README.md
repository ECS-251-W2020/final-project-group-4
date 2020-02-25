# Pytorch CUDA extension for Windows OS x64

- Using static library to avoid Pytorch bug in the [tutorial]((http://pytorch.org/tutorials/advanced/cpp_extension.html)).

# Dependencies

- Windows OS x64
- Visual Studio 2017 community
- Cuda 10.1
- Python 3.7.2
- Pytorch 1.4.0 stable

# Guide

- Make sure all the [dependencies]() are installed
- Open new CUDA runtime solution in Visual Studio named `double_kernel` in current directory
- Replace codes in `double_kernel\double_kernel\kernel.cu` with codes in [double_kernel.cu](double_kernel.cu)
- Compile the solution we have built as a static library and release version with Visual Studio
- Copy `double_kernel\x64\Release\double_kernel.lib` to current directory
- Build extension using torch.utils.cppExtension.cudaExtension in [setup.py](.\setup.py) by running code as follows

        $ python setup.py install

# Test

- Current directory
  
        $ python
- set up a tensor, turn into CUDA, do forward which we have wrapped in the new package named double_ext in [setup.py](.\setup.py).

```python
import torch
import double_ext
x = torch.tensor([1., 2., 3., 4.])
cx = x.cuda()
double_ext.forward(x)
# the output should be torch.tensor([2., 4., 6., 8.])
```
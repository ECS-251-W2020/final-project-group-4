# Group 4 Final project `'PytorchAegis'`

## Idea

We want to combine the security of hardware enclave for CPU and the speed of machine learning model for GPU, in order to manipulate data on GPU using key from CPU enclave. More details about the project can be seen in [Proposal.pdf](Proposal.pdf).

## Introduction to the directory

Three main folders:

* [AegisEngine](./AegisEngine/): Intel SGX + CUDA AES algorithm implementation
* [PytorchAegis](./PytorchAegis/): Pytorch binding for AegisEngine plus a runable demo for both array and image
* [demo](./demo/): Pytorch MNIST training with `PytorchAegis`

Other files are our previous works, such as a sample for implementing AES algorithm, a sample for Pytorch extension with CUDA, a sample for Intel Enclave(SGX).

## Build
Before building the project, you need to install following developing tools and SDK:
* Visual Studio 2017 or later
* Pytorch 1.4 (CUDA 10.1 Version)
* CUDA 10.1
* Intel SGX SDK for Windows v2.6.100.2
Of course, your CPU need to support Intel SGX and your GPU need to support CUDA.

First step, build AegisEngine with Visual Studio. Please simply open the `AegisEngine.sln` file and build in in Pre-Release mode. 

Second, copy all dll and lib files that generated in `AegisEngine/x64/Prelease` to `PytorchAegis` directory. 

Third, install PytorchAegis with following command:
```
# python setup.py install
```

Then PytorchAegis will be installed into your Python environment.

## Use cases
Please check `demo` directory.

## Future work

* How can user send key to SGX?
* Back up idea: Encrypt key by RSA algorithm. GPU generates key' pair, sends public key' to Enclave. The Enclave encrpyt key using pulic key' and send back to GPU.
<!-- * Another idea: We can implement RSA to avoid touching host memory with any true data in the whole process. GPU can generate a pair of keys. -->
* Hardware support: Communication between Enclave, protected memory and GPU memory. Or kind of protected path from outside user direct to Enclave memory.
* Enclave support: Both ecall and ocall needs a temporate copy of unprotected host memory, which is a improvable trap.





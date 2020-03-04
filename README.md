# Group 4 Final project `'PytorchAegis'`

## Idea

We want to combine the security of hardware enclave for CPU and the speed of machine learning model for GPU, in order to manipulate data on GPU using key from CPU enclave. More details about the project can be seen in [Proposal.pdf](Proposal.pdf).

## Introduction to the directory

Two main folders:

* [AegisEngine](./AegisEngine/): Intel SGX + CUDA AES algorithm implementation
* [PytorchAegis](./PytorchAegis/): Pytorch binding for AegisEngine plus a runable demo for both array and image.

Other files are our previous works, such as a sample for implementing AES algorithm, a sample for Pytorch extension with CUDA, a sample for Intel Enclave(SGX).

## Future work

* How can user send key to SGX?
* Back up idea: Encrypt key by RSA algorithm. GPU generates key' pair, sends public key' to Enclave. The Enclave encrpyt key using pulic key' and send back to GPU.
<!-- * Another idea: We can implement RSA to avoid touching host memory with any true data in the whole process. GPU can generate a pair of keys. -->
* Hardware support: Communication between Enclave, protected memory and GPU memory. Or kind of protected path from outside user direct to Enclave memory.
* Enclave support: Both ecall and ocall needs a temporate copy of unprotected host memory, which is a improvable trap.





# Group 4 Final project

## Idea

We want to combine the security of hardware enclave for CPU and the speed of machine learning model for GPU, in order to manipulate data on GPU using key from CPU enclave. More details about the project can be seen in [Proposal.pdf](Proposal.pdf).

## Introduction to the directory

Two main folders:

* [AegisEngine](./AegisEngine/): Intel SGX + CUDA AES algorithm implementation
* [PytorchAegis](./PytorchAegis/): Pytorch binding for AegisEngine plus a runable demo for both array and image.

Other files are our previous works, such as a sample for implementing AES algorithm, a sample for Pytorch extension with CUDA, a sample for Intel Enclave(SGX).

Future work: How can user send key to SGX?





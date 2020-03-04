# Weekly update 2020/02/03

## What we have done

We have build a demo for the idea in the proposal combine all the four parts in a whole, speifically, we make it run with Enclave and CUDA installed on a Windows OS computer. In demo from [PytorchAegis](./PytorchAegis/), we can use input as a array or an image to be encrypted and decrypted in CUDA memory after the key is sent there from Enlave's memory. Another [demo](./demo) using the decrypted images by our model to train a neural network almost finishes work.

## Tasks for next week

- Prepare for presentation next week
- Make some improvement of security of the whole model

## Further ideas

- We still figure out some other ideas which either improve our model or enhance the security of the communication between different parts of the machine.
  - How can user send key to SGX?
  - Back up idea: Encrypt key by RSA algorithm. GPU generates key' pair, sends public key' to Enclave. The Enclave encrpyt key using pulic key' and send back to GPU.
  - Hardware support: Communication between Enclave, protected memory and GPU memory. Or kind of protected path from outside user direct to Enclave memory.
  - Enclave support: Both ecall and ocall needs a temporate copy of unprotected host memory, which is a improvable trap.

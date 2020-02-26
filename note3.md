# What we have done

##  Divide our job into 4 parts

- Build a sample for AES algorithm which can decrypt image in GPU with CUDA.
- Build a sample for Enclave which can make memory communication with GPU's call. It will be used for saving key for AES algorithm.
- Implement AES algorithm with python which can encrypt data.
- Build a sample for Pytorch which can make CUDA extension.

# Next work

- Combine all the parts and run it under Windows OS x64 with SGX.
- It can use key stored in Enclave to decrypt encrypted image data [Cifar](https://www.cs.toronto.edu/~kriz/cifar.html) in GPU.

# Further ideas

- From last meeting with instuctor, what if current GPU is not trustworthy when the whole system is compromised?
- What can we do to make our idea still useful under assumption above? Rewrite a driver which can protect data in GPU?
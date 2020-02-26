# What we have done

##  Divide our job into 4 parts

- Built a sample for AES algorithm which can decrypt image in GPU with CUDA. (Jiemin)
- Built a sample for Enclave that can send data to device memory. It will be used for storing the key for AES algorithm. (Mengxiao)
- Implemented AES algorithm with python to encrypt data. (Chuyuan)
- Built a sample for Pytorch which can make CUDA extension. (Xiaorui)

# Tasks for next week

- Combine all the parts and run it under Windows OS x64 with SGX.
- It can use key stored in Enclave to decrypt encrypted image data [Cifar](https://www.cs.toronto.edu/~kriz/cifar.html) in GPU.

# What to expect
Next week, the enclave for storing keys and the GPU decryption algorithm will be shipped as a Pytorch extension. We will be able to test the system very soon.

# Further ideas

- From last meeting with instuctor, what if current GPU is not trustworthy when the whole system is compromised?
- What can we do to make our idea still useful under assumption above? Rewrite a driver which can protect data in GPU?

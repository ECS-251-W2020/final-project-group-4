## AES Encryption and Decryption on CUDA

* Implementation of AES Encryption and Decryption on CUDA with an image sample.
* Sample input:
  * `.bmp` format images `lena.bmp`
  * `unsigned char` encryption key
    * can with different lengths, default is 16 bytes
* Sample output:
  * `Encrypted.bmp`: the encrypted image
  * `Decrypted.bmp`: the decrypted image, same as original input.



## Dependencies

* Windows OS x64
* Visual Studio 2019 community
* Cuda 10.2



## Guide

* Make sure all the dependencies are installed.
* Open `AES.sln` with Visual Studio 2019 community and run it.
* If you want to change input:
  * **Key**: change variable `key` in `main()` of file `kernal.cu`.
    * It can be in different lengths, e.g. 16 / 20 / 24. Remember to change variable `Nk` which means the number of columns in the key.
  * **Image**: put your images in the same directory of `AES.sln`. Then change the input directory in `main()` of file `kernal.cu`.
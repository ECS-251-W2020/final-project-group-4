#pragma once
#include "constant.h"
void launchEncryptKernel(unsigned char* d_bitmapImage, unsigned char* d_expanded_key, int size);
void launchDecryptKernel(unsigned char* d_bitmapImage, unsigned char* d_expanded_key, int size);
void set_aes_key(unsigned char secret[AES_NB]);
void copy_aes_key_to_device(void* keyDevice);
int initialize_enclave(void);
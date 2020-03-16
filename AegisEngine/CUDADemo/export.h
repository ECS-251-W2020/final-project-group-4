#pragma once
#include "constant.h"
void launchEncryptKernel(unsigned char* d_bitmapImage, unsigned char* d_expanded_key, int size);
void launchDecryptKernel(unsigned char* d_bitmapImage, unsigned char* d_expanded_key, int size);
void set_aes_key(unsigned char secret[AES_NB]);
void copy_aes_key_to_device(void* keyDevice);
void copy_aes_key_to_device_rsa(void* keyDevice, int rsa_e, int rsa_n, int*rsa_private_key_device);
int initialize_enclave(void);
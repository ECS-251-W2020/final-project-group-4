#pragma once
void key_expansion(unsigned char* key, unsigned char* w);
void launchEncryptKernel(unsigned char* d_bitmapImage, unsigned char* d_expanded_key, int size);
void launchDecryptKernel(unsigned char* d_bitmapImage, unsigned char* d_expanded_key, int size);
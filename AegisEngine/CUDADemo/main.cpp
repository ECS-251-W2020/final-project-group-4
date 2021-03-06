
#include "cuda_runtime.h"
#include <stdio.h>
#include "sgx_urts.h"
#include "enclave_app.h"
#include "StorageEnclave_u.h"
#include "constant.h"
#include "aes_kernel.h"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int initialize_enclave(void);
int computeHashOnDevice(void *value);
int computeHashOnHost(unsigned char *value);
void key_expansion(unsigned char* key, unsigned char* w);

void ocall_send_to_device(unsigned char value[AES_EXP_NB], void* devicePtr) {
	cudaMemcpy(devicePtr, value, sizeof(unsigned char) * AES_EXP_NB, cudaMemcpyHostToDevice);
}

void ocall_send_to_device_rsa(int value[AES_EXP_NB], void* devicePtr) {
	cudaMemcpy(devicePtr, value, sizeof(int) * AES_EXP_NB, cudaMemcpyHostToDevice);
}


void ocall_print_secret(unsigned char value[AES_NB]) {
	for (int i = 0; i < AES_NB; ++i) {
		printf("%c", value[i]);
	}
	puts("");
}

__declspec(dllexport) void set_aes_key(unsigned char secret[AES_NB]) {
	set_secret4(global_eid, secret);
}

__declspec(dllexport) void copy_aes_key_to_device(void* keyDevice) {
	copy_secret_to_device(global_eid, keyDevice);
}

__declspec(dllexport) void copy_aes_key_to_device_rsa(void* keyDevice,int rsa_e, int rsa_n, int*rsa_private_key_device) {
	int *rsa_encrypted_key;
	cudaMalloc(&rsa_encrypted_key, AES_EXP_NB * sizeof(int));
	copy_secret_to_device_with_rsa(global_eid, &rsa_e, &rsa_n, rsa_encrypted_key);
	launchRSADecryptKernel(rsa_encrypted_key, (unsigned char*)keyDevice, rsa_private_key_device, AES_EXP_NB);
}

/*
int main()
{
	// start SGX
	if (initialize_enclave() < 0) {
		printf("Enter a character before exit ...\n");
		getchar();
		return -1;
	}
	printf("SGX Started\n");
	unsigned char secret[AES_NB] = {
		1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
	set_aes_key(secret);
	unsigned char exp_key[AES_EXP_NB];
	// only for debug
	key_expansion(secret, exp_key);
	int secret_hash = computeHashOnHost(exp_key);
	memset(exp_key, 0, sizeof(unsigned char) * AES_EXP_NB);
	memset(secret, 0, sizeof(unsigned char) * AES_NB);
	puts("Put secret into SGX and clean the host memory");
	printf("Host hash = %d\n", secret_hash);

	int rsa_e = 13, rsa_n = 437;
	int rsa_private_key[3] = { 61, 19, 23 };
	int *rsa_private_key_device;
	cudaMalloc(&rsa_private_key_device, 3 * sizeof(int));
	cudaMemcpy(rsa_private_key_device, rsa_private_key, 3 * sizeof(int), cudaMemcpyHostToDevice);
	// load expended key to GPU
	void *keyBufDevice;
	cudaMalloc(&keyBufDevice, sizeof(unsigned char) * AES_EXP_NB);
	copy_aes_key_to_device_rsa(keyBufDevice, rsa_e, rsa_n, rsa_private_key_device);
	puts("Sent secret from SGX to GPU Memory");
	int secret_device_hash = computeHashOnDevice(keyBufDevice);
	printf("device hash = %d\n", secret_device_hash);

	// generate data for encryption
	unsigned char dataBlock[2048]; 
	for (int i = 0; i < 2048; ++i) {
		dataBlock[i] = i % 200;
	}
	void *dataBufDevice; cudaMalloc(&dataBufDevice, 2048 * sizeof(unsigned char));
	cudaMemcpy(dataBufDevice, dataBlock, sizeof(unsigned char) * 2048, cudaMemcpyHostToDevice);
	launchEncryptKernel((unsigned char *)dataBufDevice, (unsigned char *)keyBufDevice, 2048);
	launchDecryptKernel((unsigned char *)dataBufDevice, (unsigned char *)keyBufDevice, 2048);
	unsigned char passData[2048];
	cudaMemcpy(passData, dataBufDevice, 2048 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(dataBufDevice);
	bool correct = true;
	for (int i = 0; i < 2048; ++i) {
		if (passData[i] != dataBlock[i]) {
			puts("Encrypt/Decrypt Error!");
			correct = false;
			break;
		}
	}
	if (correct) {
		puts("Encrypt/Decrypt Correct!");
	}

	cudaFree(keyBufDevice);
	sgx_destroy_enclave(global_eid);
	printf("Info: SampleEnclave successfully returned.\n");

	printf("Enter a character before exit ...\n");
	getchar();
	return 0;
}
*/
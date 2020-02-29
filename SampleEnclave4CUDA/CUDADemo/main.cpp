
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

void ocall_print_secret(unsigned char value[AES_NB]) {
	for (int i = 0; i < AES_NB; ++i) {
		printf("%c", value[i]);
	}
	puts("");
}
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
		0x2b, 0x28, 0xab, 0x09,
		0x7e, 0xae, 0xf7, 0xcf,
		0x15, 0xd2, 0x15, 0x4f,
		0x16, 0xa6, 0x88, 0x3c };
	set_secret4(global_eid, secret);
	unsigned char exp_key[AES_EXP_NB];
	// only for debug
	key_expansion(secret, exp_key);
	int secret_hash = computeHashOnHost(exp_key);
	memset(exp_key, 0, sizeof(unsigned char) * AES_EXP_NB);
	memset(secret, 0, sizeof(unsigned char) * AES_NB);
	puts("Put secret into SGX and clean the host memory");
	printf("Host hash = %d\n", secret_hash);

	// load expended key to GPU
	void *keyBufDevice;
	cudaMalloc(&keyBufDevice, sizeof(unsigned char) * AES_EXP_NB);
	copy_secret_to_device(global_eid, keyBufDevice);
	puts("Sent secret from SGX to GPU Memory");
	int secret_device_hash = computeHashOnDevice(keyBufDevice);
	printf("device hash = %d\n", secret_device_hash);

	// generate data for encryption
	unsigned char dataBlock[2048]; int seed = 197;
	for (int i = 0; i < 2048; ++i) {
		dataBlock[i] = seed % 256;
		seed = seed * 233 + 97;
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

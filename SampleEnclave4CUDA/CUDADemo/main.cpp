
#include "cuda_runtime.h"
#include <stdio.h>
#include "sgx_urts.h"
#include "enclave_app.h"
#include "StorageEnclave_u.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int initialize_enclave(void);
int computeHashOnDevice(void *value);
int computeHashOnHost(int *value);

void ocall_send_to_device(int value[4], void* devicePtr) {
	//memcpy(devicePtr, value, sizeof(int) * 4);
	cudaMemcpy(devicePtr, value, sizeof(int) * 4, cudaMemcpyHostToDevice);
}

void ocall_print_secret(int value[4]) {
	for (int i = 0; i < 4; ++i) {
		printf("%d", value[i]);
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
	int secret[4] = { 0x19, 0x98, 0x02, 0x09 };
	set_secret4(global_eid, secret);
	int secret_hash = computeHashOnHost(secret);
	memset(secret, 0, sizeof(int) * 4);
	puts("Put secret into SGX and clean the host memory");
	printf("Host hash = %d\n", secret_hash);
	void *bufDevice;
	cudaMalloc(&bufDevice, sizeof(int) * 4);
	copy_secret_to_device(global_eid, bufDevice);
	puts("Sent secret from SGX to GPU Memory");
	int secret_device_hash = computeHashOnDevice(bufDevice);
	printf("device hash = %d\n", secret_device_hash);
	cudaFree(bufDevice);
	sgx_destroy_enclave(global_eid);
	printf("Info: SampleEnclave successfully returned.\n");

	printf("Enter a character before exit ...\n");
	getchar();
	return 0;
}

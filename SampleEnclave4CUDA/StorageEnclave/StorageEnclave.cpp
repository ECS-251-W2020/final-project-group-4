#include "StorageEnclave_t.h"

#include "sgx_trts.h"

int my_secret_arr[4] = { 0,0,0,0};
void set_secret4(int* secret) {
	my_secret_arr[0] = secret[0];
	my_secret_arr[1] = secret[1];
	my_secret_arr[2] = secret[2];
	my_secret_arr[3] = secret[3];
}

void print_secret() {
	ocall_print_secret(my_secret_arr);
}

void copy_secret_to_device(void *devicePtr) {
	ocall_send_to_device(my_secret_arr, devicePtr);
}
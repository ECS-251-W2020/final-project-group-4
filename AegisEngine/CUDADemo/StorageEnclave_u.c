#include "StorageEnclave_u.h"
#include <errno.h>

typedef struct ms_set_secret4_t {
	unsigned char* ms_secret;
} ms_set_secret4_t;

typedef struct ms_copy_secret_to_device_t {
	void* ms_devicePtr;
} ms_copy_secret_to_device_t;

typedef struct ms_copy_secret_to_device_with_rsa_t {
	int* ms_rsa_e;
	int* ms_rsa_n;
	void* ms_devicePtr;
} ms_copy_secret_to_device_with_rsa_t;

typedef struct ms_ocall_print_secret_t {
	unsigned char* ms_value;
} ms_ocall_print_secret_t;

typedef struct ms_ocall_send_to_device_t {
	unsigned char* ms_value;
	void* ms_devicePtr;
} ms_ocall_send_to_device_t;

typedef struct ms_ocall_send_to_device_rsa_t {
	int* ms_value;
	void* ms_devicePtr;
} ms_ocall_send_to_device_rsa_t;

typedef struct ms_sgx_oc_cpuidex_t {
	int* ms_cpuinfo;
	int ms_leaf;
	int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
	int ms_retval;
	const void* ms_waiter;
	const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
	int ms_retval;
	const void** ms_waiters;
	size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

static sgx_status_t SGX_CDECL StorageEnclave_ocall_print_secret(void* pms)
{
	ms_ocall_print_secret_t* ms = SGX_CAST(ms_ocall_print_secret_t*, pms);
	ocall_print_secret(ms->ms_value);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_ocall_send_to_device(void* pms)
{
	ms_ocall_send_to_device_t* ms = SGX_CAST(ms_ocall_send_to_device_t*, pms);
	ocall_send_to_device(ms->ms_value, ms->ms_devicePtr);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_ocall_send_to_device_rsa(void* pms)
{
	ms_ocall_send_to_device_rsa_t* ms = SGX_CAST(ms_ocall_send_to_device_rsa_t*, pms);
	ocall_send_to_device_rsa(ms->ms_value, ms->ms_devicePtr);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_sgx_oc_cpuidex(void* pms)
{
	ms_sgx_oc_cpuidex_t* ms = SGX_CAST(ms_sgx_oc_cpuidex_t*, pms);
	sgx_oc_cpuidex(ms->ms_cpuinfo, ms->ms_leaf, ms->ms_subleaf);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_sgx_thread_wait_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_wait_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_wait_untrusted_event_ocall(ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_sgx_thread_set_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_set_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_untrusted_event_ocall(ms->ms_waiter);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_sgx_thread_setwait_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_setwait_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_setwait_untrusted_events_ocall(ms->ms_waiter, ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL StorageEnclave_sgx_thread_set_multiple_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_multiple_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_multiple_untrusted_events_ocall(ms->ms_waiters, ms->ms_total);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * func_addr[8];
} ocall_table_StorageEnclave = {
	8,
	{
		(void*)(uintptr_t)StorageEnclave_ocall_print_secret,
		(void*)(uintptr_t)StorageEnclave_ocall_send_to_device,
		(void*)(uintptr_t)StorageEnclave_ocall_send_to_device_rsa,
		(void*)(uintptr_t)StorageEnclave_sgx_oc_cpuidex,
		(void*)(uintptr_t)StorageEnclave_sgx_thread_wait_untrusted_event_ocall,
		(void*)(uintptr_t)StorageEnclave_sgx_thread_set_untrusted_event_ocall,
		(void*)(uintptr_t)StorageEnclave_sgx_thread_setwait_untrusted_events_ocall,
		(void*)(uintptr_t)StorageEnclave_sgx_thread_set_multiple_untrusted_events_ocall,
	}
};

sgx_status_t set_secret4(sgx_enclave_id_t eid, unsigned char secret[16])
{
	sgx_status_t status;
	ms_set_secret4_t ms;
	ms.ms_secret = (unsigned char*)secret;
	status = sgx_ecall(eid, 0, &ocall_table_StorageEnclave, &ms);
	return status;
}

sgx_status_t print_secret(sgx_enclave_id_t eid)
{
	sgx_status_t status;
	status = sgx_ecall(eid, 1, &ocall_table_StorageEnclave, NULL);
	return status;
}

sgx_status_t copy_secret_to_device(sgx_enclave_id_t eid, void* devicePtr)
{
	sgx_status_t status;
	ms_copy_secret_to_device_t ms;
	ms.ms_devicePtr = devicePtr;
	status = sgx_ecall(eid, 2, &ocall_table_StorageEnclave, &ms);
	return status;
}

sgx_status_t copy_secret_to_device_with_rsa(sgx_enclave_id_t eid, int* rsa_e, int* rsa_n, void* devicePtr)
{
	sgx_status_t status;
	ms_copy_secret_to_device_with_rsa_t ms;
	ms.ms_rsa_e = rsa_e;
	ms.ms_rsa_n = rsa_n;
	ms.ms_devicePtr = devicePtr;
	status = sgx_ecall(eid, 3, &ocall_table_StorageEnclave, &ms);
	return status;
}


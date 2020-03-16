#ifndef STORAGEENCLAVE_U_H__
#define STORAGEENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */


#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_PRINT_SECRET_DEFINED__
#define OCALL_PRINT_SECRET_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print_secret, (unsigned char value[16]));
#endif
#ifndef OCALL_SEND_TO_DEVICE_DEFINED__
#define OCALL_SEND_TO_DEVICE_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_send_to_device, (unsigned char value[176], void* devicePtr));
#endif
#ifndef OCALL_SEND_TO_DEVICE_RSA_DEFINED__
#define OCALL_SEND_TO_DEVICE_RSA_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_send_to_device_rsa, (int value[176], void* devicePtr));
#endif
#ifndef SGX_OC_CPUIDEX_DEFINED__
#define SGX_OC_CPUIDEX_DEFINED__
void SGX_UBRIDGE(SGX_CDECL, sgx_oc_cpuidex, (int cpuinfo[4], int leaf, int subleaf));
#endif
#ifndef SGX_THREAD_WAIT_UNTRUSTED_EVENT_OCALL_DEFINED__
#define SGX_THREAD_WAIT_UNTRUSTED_EVENT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_wait_untrusted_event_ocall, (const void* self));
#endif
#ifndef SGX_THREAD_SET_UNTRUSTED_EVENT_OCALL_DEFINED__
#define SGX_THREAD_SET_UNTRUSTED_EVENT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_set_untrusted_event_ocall, (const void* waiter));
#endif
#ifndef SGX_THREAD_SETWAIT_UNTRUSTED_EVENTS_OCALL_DEFINED__
#define SGX_THREAD_SETWAIT_UNTRUSTED_EVENTS_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_setwait_untrusted_events_ocall, (const void* waiter, const void* self));
#endif
#ifndef SGX_THREAD_SET_MULTIPLE_UNTRUSTED_EVENTS_OCALL_DEFINED__
#define SGX_THREAD_SET_MULTIPLE_UNTRUSTED_EVENTS_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_set_multiple_untrusted_events_ocall, (const void** waiters, size_t total));
#endif

sgx_status_t set_secret4(sgx_enclave_id_t eid, unsigned char secret[16]);
sgx_status_t print_secret(sgx_enclave_id_t eid);
sgx_status_t copy_secret_to_device(sgx_enclave_id_t eid, void* devicePtr);
sgx_status_t copy_secret_to_device_with_rsa(sgx_enclave_id_t eid, int* rsa_e, int* rsa_n, void* devicePtr);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

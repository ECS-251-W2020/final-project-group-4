#include <torch/extension.h>
#include <vector>
#include <iostream>
#include "export.h"
/*
std::vector<at::Tensor> double_ext_forward(torch::Tensor input) {
    auto ret = at::add(input, input);
    return {ret};
}

std::vector<at::Tensor> double_ext_backward(torch::Tensor grad_y) {
    auto ret = at::add(grad_y, grad_y);
    return {ret};
}*/

/*
at::Tensor double_ext_forward(torch::Tensor input) {
    auto ret = at::zeros_like(input);
    const int state_size = input.size(0);
    
    //AT_DISPATCH_FLOATING_TYPES(input.type(), "double_ext_forward", ([&] {
    //    double_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(), ret.data<scalar_t>(), state_size);
    //}));
    launch_double_ext_cuda_kernel(input.data<unsigned char>(), ret.data<unsigned char>(), state_size);  // need statement
    return ret;
}

at::Tensor double_ext_backward(torch::Tensor grad_y) {
    auto ret = at::add(grad_y, grad_y);
    return ret;
}*/
int set_aegis_key(torch::Tensor key_tensor) {
    const int state_size = key_tensor.size(0);
    if (state_size != AES_NB) {
		return -1;
	}
	set_aes_key(key_tensor.data<unsigned char>());
	return 0;
}

at::Tensor get_aegis_key_cuda() {
	auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA, 0);
	auto output = at::zeros({AES_EXP_NB}, options);
    copy_aes_key_to_device(output.data<unsigned char>());
    return output;
}

at::Tensor encrypt_data(torch::Tensor input, torch::Tensor key) {
	auto ret = input.clone().contiguous();
	const int size = ret.size(0);
	launchEncryptKernel(ret.data<unsigned char>(), key.data<unsigned char>(), size);
    return ret;
}
at::Tensor decrypt_data(torch::Tensor input, torch::Tensor key) {
	auto ret = input.clone();
	const int size = ret.size(0);
	launchDecryptKernel(ret.data<unsigned char>(), key.data<unsigned char>(), size);
    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize_enclave", &initialize_enclave, "Initializing Intel SGX Enclave");
  m.def("set_aegis_key", &set_aegis_key, "Set 16-length key for Aegis");
  m.def("get_aegis_key_cuda", &get_aegis_key_cuda, "Get expended key on CUDA");
  m.def("encrypt_data", &encrypt_data, "Encrypy an uint8 1-D tensor.");
  m.def("decrypt_data", &decrypt_data, "Decrypy an uint8 1-D tensor.");
}
#include <torch/extension.h>
#include <vector>
#include <iostream>
/*
std::vector<at::Tensor> double_ext_forward(torch::Tensor input) {
    auto ret = at::add(input, input);
    return {ret};
}

std::vector<at::Tensor> double_ext_backward(torch::Tensor grad_y) {
    auto ret = at::add(grad_y, grad_y);
    return {ret};
}*/

void launch_double_ext_cuda_kernel(const float* data, float* output, size_t n);

at::Tensor double_ext_forward(torch::Tensor input) {
    auto ret = at::zeros_like(input);
    const int state_size = input.size(0);
    
    //AT_DISPATCH_FLOATING_TYPES(input.type(), "double_ext_forward", ([&] {
    //    double_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(), ret.data<scalar_t>(), state_size);
    //}));
    launch_double_ext_cuda_kernel(input.data<float>(), ret.data<float>(), state_size);
    return ret;
}

at::Tensor double_ext_backward(torch::Tensor grad_y) {
    auto ret = at::add(grad_y, grad_y);
    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &double_ext_forward, "DoubleExt forward");
  m.def("backward", &double_ext_backward, "DoubleExt backward");
}
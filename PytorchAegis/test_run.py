import torch
import pytorch_aegis
x = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], dtype=torch.uint8)
pytorch_aegis.initialize_enclave()
pytorch_aegis.set_aegis_key(x)
key = pytorch_aegis.get_aegis_key_cuda()
data = torch.randint(0, 200, (2048,), dtype=torch.uint8).cuda()
for i in range(2048):
    data[i] = i % 200
edata = pytorch_aegis.encrypt_data(data, key)
dedata = pytorch_aegis.decrypt_data(edata, key)
print("key", key)
print("data", data)
print("edata", edata)
print("dedata", dedata)
print("difference = ", (data - dedata).sum())
from IPython import embed; embed()
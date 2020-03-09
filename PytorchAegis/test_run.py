import torch
import pytorch_aegis
x = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], dtype=torch.uint8) # Key to encrpt data
pytorch_aegis.initialize_enclave()
pytorch_aegis.set_aegis_key(x)
key = pytorch_aegis.get_aegis_key_cuda()

# Demo for an array

data = torch.randint(0, 200, (2048,), dtype=torch.uint8).cuda()
for i in range(2048):
    data[i] = i % 200
edata = pytorch_aegis.encrypt_data(data, key)
dedata = pytorch_aegis.decrypt_data(edata, key)


# # Demo for an image
# import image_to_numpy
# data = image_to_numpy.load_image_file("my_file.jpg")
# data = torch.as_tensor(data)
# datasize = data.size()
# # data = data.reshape(-1).cuda()
# data = data.cuda()   # The pytorch_aegis is compatible with normal matrix, not just for array
# edata = pytorch_aegis.encrypt_data(data, key)
# dedata = pytorch_aegis.decrypt_data(edata, key)

# # Save the decrypted data of image for comparison
# imgt = torch.tensor(dedata,device = 'cpu')
# imgt = imgt.reshape(datasize)
# import matplotlib.pyplot as plt
# plt.imshow(imgt)
# plt.savefig(fname = 'my_file_de.jpg', format = 'jpg')

# Print the result 
print("key", key)
print("data", data)
print("edata", edata)
print("dedata", dedata)
print("difference = ", (data - dedata).abs().sum())
# from IPython import embed; embed()
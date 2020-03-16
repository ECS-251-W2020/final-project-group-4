import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import time
import pytorch_aegis

epochs = 100



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def load_and_encrypt(key, is_trainning):
    dataset = datasets.MNIST('./data', train=is_trainning)
    # encrypted_data = torch.zeros([dataset.__len__(), 28 * 28], device='cuda')
    encrypted_data = torch.zeros([6400, 28 * 28], device='cuda', dtype=torch.uint8)
    # only use 640 samples for debugging
    for idx, sample in enumerate(dataset.data[:6400].to('cuda')):
        encrypted_data[idx] = pytorch_aegis.encrypt_data(sample.flatten(), key)
    labels = dataset.targets.to('cuda')
    return encrypted_data, labels


def decrypt_and_normalize(data, key):
    # decrypt data
    tmp = torch.zeros([len(data), 28 * 28], device='cuda', dtype=torch.uint8)
    for i, sample in enumerate(data):
        decrypted_sample = pytorch_aegis.decrypt_data(sample, key)
        decrypted_sample = decrypted_sample
        tmp[i] = decrypted_sample

    # normalize data
    tmp = tmp.float() / 255
    means = torch.mean(tmp)
    stds = torch.std(tmp)
    decrypted_data = torch.zeros([len(data), 1, 28, 28], device='cuda', dtype=torch.float)
    decrypted_data = (tmp.reshape([len(data), 1, 28, 28]) - means) / stds
    return decrypted_data


if __name__ == '__main__':
    # key setting
    key = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.uint8)
    pytorch_aegis.initialize_enclave()
    pytorch_aegis.set_aegis_key(key)
    key = pytorch_aegis.get_aegis_key_cuda()

    start_time = time.time()
    # load encrypted data
    encrypted_train_data, train_labels = load_and_encrypt(key, True)
    encrypted_test_data, test_labels = load_and_encrypt(key, False)

    # decrypt data
    decrypted_train_data = decrypt_and_normalize(encrypted_train_data, key)
    decrypted_test_data = decrypt_and_normalize(encrypted_test_data, key)
    end_time = time.time()
    data_loading_time = end_time - start_time
    # define model
    net = Net()
    net = net.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    for epoch in range(epochs):
        # train
        running_loss = 0.0
        for i in range(int(len(decrypted_train_data) / 64)):
            # batch samples
            batch = decrypted_train_data[i * 64: (i + 1) * 64]
            labels = train_labels[i * 64: (i + 1) * 64]
            # run
            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        # test
        correct_num = 0.0
        for i in range(int(len(decrypted_test_data) / 64)):
            # batch samples
            batch = decrypted_test_data[i * 64: (i + 1) * 64]
            labels = test_labels[i * 64: (i + 1) * 64]
            # run
            optimizer.zero_grad()
            outputs = net(batch)
            _, predict = torch.max(outputs, 1)
            correct_num += torch.sum(predict == labels)

        print("Test Accuracy =", correct_num.item() / len(encrypted_test_data))

    end_time = time.time()
    print('Finished Training')
    print("data loading cost = ", data_loading_time, 's')
    print("training time cost =", end_time - start_time, 's')

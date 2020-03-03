import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import time
import pytorch_aegis

epochs = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, key):
        x = pytorch_aegis.decrypt_data(x, key)
        x = x.reshape((1, 28, 28))
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


def load_encrypted_data(key):
    # train data
    train_dataset = datasets.MNIST('./data', train=True)
    encrypted_train_data = []
    train_labels = train_dataset.targets
    for data in train_dataset.data.to('cuda'):
        encrypted_data = pytorch_aegis.encrypt_data(data.flatten(), key)
        encrypted_train_data.append(encrypted_data)

    # test data
    test_dataset = datasets.MNIST('./data', train=False)
    encrypted_test_data = []
    test_labels = test_dataset.targets
    for data in test_dataset.data.to('cuda'):
        encrypted_data = pytorch_aegis.encrypt_data(data.flatten(), key)
        encrypted_test_data.append(encrypted_data)
    return encrypted_train_data.to('cpu'), train_labels, encrypted_test_data.to('cpu'), test_labels


if __name__ == '__main__':
    # key setting
    key = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.uint8)
    pytorch_aegis.initialize_enclave()
    pytorch_aegis.set_aegis_key(key)
    key = pytorch_aegis.get_aegis_key_cuda()

    # load encrypted data
    encrypted_train_data, train_labels, encrypted_test_data, test_labels = load_encrypted_data(key)

    # define model
    net = Net()
    net = net.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    for epoch in range(epochs):
        # train
        running_loss = 0.0
        for i in range(int(len(encrypted_train_data) / 64)):
            # batch samples
            inputs = encrypted_train_data[i * 64: (i + 1) * 64]
            labels = train_labels[i * 64: (i + 1) * 64]
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # run
            optimizer.zero_grad()
            outputs = net(inputs, key)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        # test
        correct_num = 0.0
        for i in range(int(len(encrypted_test_data) / 64)):
            # batch samples
            inputs = encrypted_test_data[i * 64: (i + 1) * 64]
            labels = test_labels[i * 64: (i + 1) * 64]
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # run
            optimizer.zero_grad()
            outputs = net(inputs, key)
            _, predict = torch.max(outputs, 1)
            correct_num += torch.sum(predict == labels)

        print("Test Accuracy =", correct_num.item() / len(encrypted_test_data))

    end_time = time.time()
    print('Finished Training')
    print("time cost =", end_time - start_time, 's')

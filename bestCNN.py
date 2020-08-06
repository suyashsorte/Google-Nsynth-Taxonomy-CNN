import torch
import torchvision
import torchvision.datasets as dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data as data
from random import shuffle
from pytorch_nsynth.nsynth import NSynth
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import wave
import sys

import pandas as pd

transform = transforms.Compose(
    [transforms.Lambda(lambda x: x / np.iinfo(np.int16).max),
     transforms.Lambda(lambda x: torch.from_numpy(x).float()), transforms.Lambda(lambda x: x[0:64000])])
train_dataset = NSynth(
    "/local/sandbox/nsynth/nsynth-train",
    transform=transform,
    blacklist_pattern=["synth_lead"],  # blacklist string istrument
    categorical_field_list=["instrument_family", "instrument_source"])
# print(type(train_dataset))

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = NSynth(
    "/local/sandbox/nsynth/nsynth-test",
    transform=transform,
    blacklist_pattern=["synth_lead"],  # blacklist string instrument
    categorical_field_list=["instrument_family", "instrument_source"])
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)
loss_validation = []
loss_train = []
valid_dataset = NSynth(
    "/local/sandbox/nsynth/nsynth-valid",
    transform=transform,
    blacklist_pattern=["synth_lead"],  # blacklist string instrument
    categorical_field_list=["instrument_family", "instrument_source"])

valid_loader = data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print(device)
plt.figure()
# print(test_dataset[8][1],'----------')
# print(test_dataset[8][0].numpy().shape)
# print(test_dataset[8][0].numpy())

# plt.plot((test_dataset[8][0]).cpu().numpy())
# plt.show()
for i in range(10):
    plt.plot((test_dataset[i][0]).cpu().numpy())
    plt.show()
classes = ('bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal')


class NetSimple(nn.Module):
    def __init__(self):
        super(NetSimple, self).__init__()
        self.fc3 = nn.Linear(255952, 10)

    def forward(self, x):
        x = self.fc3(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(255952, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool1d(F.leaky_relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net.load_state_dict(torch.load("dict_model.pwf"))
net.to(device)

target = []


def train_data(train_loader):
    net.train()

    for sample, instrument_family, instrument_source_target, targets in train_loader:
        # sample = sample[0:20]
        sample, instrument_family = sample.to(device), instrument_family.to(device)
        sample = sample.unsqueeze(1)  # -----------------------
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
        # optimizer = optim.Adam(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = net(sample)
        loss = F.cross_entropy(output, instrument_family)
        loss.backward()
        optimizer.step()
    loss_train.append(loss)


pred = []


def validation():
    net.eval()
    val_loss = 0
    correct = 0
    val_len = len(valid_loader.dataset)
    output = 0
    for sample, instrument_family, instrument_source_target, targets in valid_loader:
        # sample = sample[0:20]
        sample, instrument_family = sample.to(device), instrument_family.to(device)
        sample = sample.unsqueeze(1)
        output = net(sample)
        val_loss += F.cross_entropy(output, instrument_family, size_average=False).data[0]
        predict = output.data.max(1, keepdim=True)[1]
        correct += predict.eq(instrument_family.data.view_as(predict)).cpu().sum()
        target.append(instrument_family)
        pred.append(predict)
    # print(output)
    val_loss /= val_len
    loss_validation.append(val_loss)
    print('Validation Average loss: {:.4f}'.format(val_loss))
    print("Accuracy: {}/{} ({:.2f}%)".format(correct, val_len, 100. * correct / val_len))


confusion_matrix = [[0. for i in range(10)] for j in range(10)]
class_accuracy = {}
one_d = [0. for i in range(10)]
two_d = [0. for i in range(10)]
list_of_index = [0. for i in range(10)]
list_of_index2 = [0. for i in range(10)]


def test():
    net.eval()
    val_loss = 0
    correct = 0
    val_len = len(test_loader.dataset)
    for sample, instrument_family, instrument_source_target, targets in test_loader:
        # sample = sample[0:20]

        sample, instrument_family = sample.to(device), instrument_family.to(device)
        sample = sample.unsqueeze(1)
        output = net(sample)
        val_loss += F.cross_entropy(output, instrument_family, size_average=False).data[0]
        predict = output.data.max(1, keepdim=True)[1]
        correct += predict.eq(instrument_family.data.view_as(predict)).cpu().sum()
        # print(output)
        # prob = np.asarray(output)
        # b = max(i)
        # for i in prob:
        #     for j in range(10):
        #         y = np.exp(i[j] - b)
        #         print(y/y.sum())
        np_output = (output).cpu()
        ar_output = (output).data.cpu().numpy()
        sorted = []
        for i in range(len(np_output)):
            maximum, index = torch.max(np_output[i], 0)
            np_output[i][index] = -100000000
            second_max, index2 = torch.max(np_output[i], 0)
            # print(maximum,index)
            # print(second_max,index2)
            a = instrument_family[i].cpu()
            if a == index:
                #    print("ent")
                # print("enter")
                one_d[a] = sample[index][0].data.cpu().numpy()
                list_of_index[a] = index.data
                two_d[a] = sample[index2][0].data.cpu().numpy()
                list_of_index2[a] = index2.data

            # print('a',a)
            # print(index)
            # print(sample[index].data)
            # print(sample[index][0].data.cpu().numpy())

            # plt.plot(sample[index][0].data.cpu().numpy())
            # plt.show()
            # print('index:', index, 'maximum:', maximum)
            # print('index:',index,'maximum:',maximum)
        # print(output,'---')
        for i in range(len(instrument_family)):
            confusion_matrix[instrument_family[i]][predict[i]] += 1
        target.append(instrument_family)
        pred.append(predict)
    val_loss /= val_len
    # loss_validation.append(val_loss)

    print('Validation Average loss: {:.4f}'.format(val_loss))
    print("Accuracy: {}/{} ({:.2f}%)".format(correct, val_len, 100. * correct / val_len))
    if i not in class_accuracy:
        class_accuracy[i] = 100. * correct / val_len
    else:
        class_accuracy[i] += 100. * correct / val_len


def learning_curve():
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.plot(loss_train, label="Train")
    plt.plot(loss_validation, label="Validation")
    plt.legend()
    # plt.show()
    plt.savefig("learning curve for MNIST.png")
    plt.close()


if __name__ == '__main__':
    epoch = 8
    # for e in range(epoch):
    #    print(e + 1)
    #    train_data(train_loader)
    #    validation()
    # torch.save(net.state_dict(), "dict_model.pwf")
    test()
    learning_curve()
    pitch = []
    for i in confusion_matrix:
        print(i)

    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
    plt.savefig("confusion matrix.png")
    plt.close()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for sample, instrument_family, instrument_source_target, targets in test_loader:
        sample, instrument_family = sample.to(device), instrument_family.to(device)
        sample = sample.unsqueeze(1)
        # output = net(sample)
        outputs = net(sample)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == instrument_family).squeeze()
        for i in range(4):
            label = instrument_family[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    j = 0
    for i, k in zip(one_d, two_d):
        print("for correct class ", classes[list_of_index[j]], "tensor:", list_of_index[j])
        plt.plot(i)
        plt.show()
        print("for near correct class ", classes[list_of_index2[j]], "tensor:", list_of_index2[j])
        plt.plot(k)
        plt.show()
        # i=i.squeeze(1)
        j = j + 1

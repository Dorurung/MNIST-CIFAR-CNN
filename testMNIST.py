import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
from torch.nn.init import kaiming_uniform_
from torch.utils.data.sampler import SubsetRandomSampler

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        global variance
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 50*4*4)
        x = self.fc1(x)
        if variance == 'sigmoid':
            x = torch.sigmoid(x)
        elif variance == 'tanh':
            x = torch.tanh(x)
        elif variance == 'relu':
            x = F.relu(x)
        else:
            x = F.relu(x)
        x = self.fc2(x)
        if type(variance) is not int and variance[:7] == 'sigmoid' and variance != 'sigmoid':
            x = torch.sigmoid(x)
        elif type(variance) is not int and variance[:7] == 'softmax':
            x = F.log_softmax(x, dim=1)
        else:
            x = F.log_softmax(x, dim=1)
        return x

variance = 'best'
if variance == 'best':
    batch_size = 4
else:
    batch_size = 128

mnist_test = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

f = open(f"best test result.txt", 'w')

net = Net()
net.load_state_dict(torch.load("model/mnist_best.pt"))
net.to(device)
net.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print(f'test Accuracy 10000 test images = {(100 * correct/total)}%')

f.write(f"best {variance}:\n")
f.write(f" Test Accuracy: {float(correct)/total}\n")
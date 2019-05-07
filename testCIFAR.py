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
from torch.utils.data.sampler import SubsetRandomSampler

'''
modified to fit dataset size
'''
NUM_CLASSES = 10

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

num_epochs = 20
batch_size = 128
learning_rate = 0.01
dropout_rate = 0.1

dropout_rate = input("Dropout rate를 입력해주세요(0~1): ")
dropout_rate = float(dropout_rate)

cifar_test = dsets.CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
alexnet = AlexNet()
alexnet.load_state_dict(torch.load(f"model/cifar_{dropout_rate}.pt"))
alexnet = alexnet.to(device)

test_loader = torch.utils.data.DataLoader(dataset=cifar_test, batch_size=batch_size, shuffle=False)
f = open(f"cifar dropout{dropout_rate} test result.txt", 'w')
alexnet.eval()
correct = [0,0,0,0,0,0,0,0,0,0]
total = [0,0,0,0,0,0,0,0,0,0]
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = alexnet(images)
    _, predicted = torch.max(outputs.data, 1)
    for j in range(labels.shape[0]):
        total[labels[j]] += 1
    for j in range(labels.shape[0]):
        if predicted[j] == labels[j]:
            correct[labels[j]] += 1

f.write(f"Final Correct/Total:\n")
f.write(f" Test Accuracy: {correct, total}\n")

#tuple = accuracy_over_epoch['train'][-1]
#tuple2 = accuracy_over_epoch['val'][-1]

f.write(f"Final Accuracy:\n")
f.write(f" Test Accuracy: {np.array(correct)/np.array(total)}{(np.array(correct)/np.array(total)).mean()}\n")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
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
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

num_epochs = 5
batch_size = 128
learning_rate = 0.001

cifar_train = dsets.CIFAR10(root='data/', train=True, transform=transforms.ToTensor(), download=True)
cifar_test = dsets.CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)

validation_split = 0.16666
dataset_len = len(cifar_train)
indices = list(range(dataset_len))
val_len = int(np.floor(validation_split * dataset_len))
validation_idx = np.random.choice(indices, size=val_len, replace=False)
train_idx = list(set(indices) - set(validation_idx))
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=batch_size, sampler=validation_sampler)
data_loaders = {"train": train_loader, "val": validation_loader}
data_lengths = {"train": len(train_idx), "val": val_len}
test_loader = torch.utils.data.DataLoader(dataset=cifar_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
alexnet = AlexNet()
alexnet = alexnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=learning_rate)
accuracy_over_epoch = {'train': [], 'val': []}

f = open("result.txt", 'w')

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            alexnet.train(True)
        else:
            alexnet.train(False)
        correct = [0,0,0,0,0,0,0,0,0,0]
        total = [0,0,0,0,0,0,0,0,0,0]
        for i, (images, labels) in enumerate(data_loaders[phase]):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = alexnet(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            for j in range(labels.shape[0]):
                total[labels[j]] += 1
            for j in range(labels.shape[0]):
                if predicted[j] == labels[j]:
                    correct[labels[j]] += 1
            if phase == 'train':
                loss.backward()
                optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(data_loaders[phase].dataset)/batch_size}] Loss: {loss.data.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], {phase} Accuracy: {float(sum(correct))/sum(total)}')
        accuracy_over_epoch[phase].append((correct, total))

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
f.write(f" Train Accuracy: {accuracy_over_epoch['train'][-1]}\n")
f.write(f" Validation Accuracy: {accuracy_over_epoch['val'][-1]}\n")
f.write(f" Test Accuracy: {correct, total}\n")

tuple = accuracy_over_epoch['train'][-1]
tuple2 = accuracy_over_epoch['val'][-1]


f.write(f"Final Accuracy:\n")
f.write(f" Train Accuracy: {np.array(tuple[0])/np.array(tuple[1])}\n")
f.write(f" Validation Accuracy: {np.array(tuple2[0])/np.array(tuple2[1])}\n")
f.write(f" Test Accuracy: {np.array(correct)/np.array(total)}\n")
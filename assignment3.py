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

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(50 * 4 * 4, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(x, kernel_size=2, stride=2)
		x = self.conv2(x)
		x = F.max_pool2d(x, kernel_size=2, stride=2)
		x = x.view(-1, 50*4*4)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

net = Net()

num_epochs = 5
batch_size = 100
learning_rate = 0.001

mnist_train = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

validation_split = 0.1
dataset_len = len(mnist_train)
indices = list(range(dataset_len))
val_len = int(np.floor(validation_split * dataset_len))
validation_idx = np.random.choice(indices, size=val_len, replace=False)
train_idx = list(set(indices) - set(validation_idx))
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, sampler=validation_sampler)
data_loaders = {"train": train_loader, "val": validation_loader}
data_lengths = {"train": len(train_idx), "val": val_len}
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
accuracy_over_epoch = {'train': [], 'val': []}

for epoch in range(num_epochs):
	for phase in ['train', 'val']:
		if phase == 'train':
			net.train(True)
		else:
			net.train(False)
		total = correct = 0
		for i, (images, labels) in enumerate(data_loaders[phase]):
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net(images)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
			if phase == 'train':
				loss.backward()
				optimizer.step()
			if (i+1) % 10 == 0:
				print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(data_loaders[phase].dataset)/batch_size}] Loss: {loss.data.item():.4f}')
		print(f'Epoch [{epoch+1}/{num_epochs}], {phase} Accuracy: {float(correct)/total}')
		accuracy_over_epoch[phase].append(float(correct)/total)

net.eval()
correct = 0
total = 0
for images, labels in test_loader:
	images = images.to(device)
	outputs = net(images)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()
print(f'test Accuracy 10000 test images = {(100 * correct/total)}%')

#net.to(device)
#inputs, labels = inputs.to(device), labels.to(device)
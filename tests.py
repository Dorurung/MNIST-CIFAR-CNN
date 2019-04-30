import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


#Define Class
class NNModel(torch.nn.Module):
    def __init__(self,act7th="tanh",act9th="softmax"):
        super(NNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size = 5) # Layer 2
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size = 5) # Layer 4
        self.fc1 = torch.nn.Linear(50*4*4,500) #Layer 6
        self.fc2 = torch.nn.Linear(500,10) # Layer 8
        self.act7th = act7th;
        self.act9th = act9th;

    def forward(self, x):
        x = F.relu(self.conv1(x)) # Layer 2
        x = F.max_pool2d(x, 2, 2) # Layer 3
        x = F.relu(self.conv2(x)) # Layer 4
        x = F.max_pool2d(x, 2, 2) # Layer 5
        x = x.view(-1, 50*4*4)  # Reshape tensor for connecting with Layer 6

        # Layer 6 & 7
        if self.act7th == "tanh":
            x = torch.tanh(self.fc1(x)) 
        elif self.act7th == "relu":
            x = F.relu(self.fc1(x))
        elif self.act7th == "sigmoid":
            x = torch.sigmoid(self.fc1(x)) 

        # Layer 8 & 9
        if self.act9th == "softmax":
            x = F.log_softmax(self.fc2(x), dim=1) 
        elif self.act9th == "sigmoid":
            x = torch.sigmoid(self.fc2(x)) 

        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Set parameters ##
batch_size = 32                          # Sizes of Mini-batch
activation_7th = "relu"                  # Activation function in 7th layer
activation_9th = "sigmoid"               # Activation function in 9th layer
criterion_t = "CE"                       # Loss Functions L1, L2, CE
init = 0                                 # 0:Uniform 1: gaussian 2: Xavier 3: MSRA
  

print("Batch size = ",batch_size)
print("Activation Function of 7th Layer =",activation_7th)
print("Activation Function of 9th Layer =",activation_9th)
print("Loss function:",criterion_t)
print("Initialization:",init)

# Model Declaration 
model = NNModel(activation_7th,activation_9th)

# Loss function Declaration
if criterion_t == "L1":
    criterion = torch.nn.L1Loss
elif criterion_t == "L2":
    criterion = torch.nn.MSELoss
else:
    criterion = torch.nn.CrossEntropyLoss()  

# Weight Initializations
for param in model.parameters():
    if param.dim() > 1 : # Skip activation function's weight
        if init == 0: 
            torch.nn.init.uniform_(param,a= -1,b = 1)
        elif init == 1:
            torch.nn.init.normal_(param,mean = 0, std = 1)
        elif init == 2:
            torch.nn.init.xavier_uniform_(param)
        elif(init == 3):
            torch.nn.init.kaiming_uniform_(param)
       

 
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(),lr = 0.005, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
 
model= model.to(device)
 
# MNIST Dataset
train_dataset = datasets.MNIST(root='data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
 
test_dataset = datasets.MNIST(root='data/',
                              train=False,
                              transform=transforms.ToTensor())
 
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
 
def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
         
        data = data.to(device)
        target = target.to(device)
         
        output = model(data)
 
        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
         
        if batch_idx%50==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data))
             
def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:
         
        data = data.to(device)
        target = target.to(device)
         
        output = model(data)
         
        test_loss += criterion(output,target).data
         
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
         
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
             
             
for epoch in range(1, 9):
    train(epoch)
    test()
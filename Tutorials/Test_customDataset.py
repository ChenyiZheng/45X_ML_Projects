import torch
import torchvision
import torchvision.transforms as transforms
# from torch.utils.data import

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn               # neural network
import torch.nn.functional as F
import torch.optim as optim         # loss function and optimizer

from Customized_dataset.customDataset import PrepareDataset

batch_size = 1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomCrop((32, 32), padding=4), # resize image to be consistent with the CIFAR10 CNN parameters
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image_object = PrepareDataset(csv_file=r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_data.csv',
                              root_dir=r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_dataset',
                              transform=transform)

# # 1.A Load data from customized dataset
# image_object = PrepareDataset(csv_file=r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_data.csv',
#                               root_dir=r'E:\cheny\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_dataset',
#                               transform=transforms.ToTensor())

# print(len(image_object))  # debug, get total length of the dataset

trainset, testset = torch.utils.data.random_split(image_object, [round(0.8*len(image_object)), len(image_object)-round(0.8*len(image_object))])
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
classes = ('fire', 'non-fire')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormaliz
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
# show images
imshow(torchvision.utils.make_grid(images))

# 2. Define a convolutional neural network that takes 3-channel image
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input image channel (RGB type), 6 output channels, 5x5 square convolution (kernel = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5*5 from image dimension, input: 16*5*5, output: 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# 3. Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the Network
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # debug
        print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save the trained model
PATH = './customized_fire_dataset.pth'
torch.save(net.state_dict(), PATH)

# 5. Test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()
# print images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
imshow(torchvision.utils.make_grid(images))

net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(batch_size)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
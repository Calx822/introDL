from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR



input_dim=784 #Mnist: 28x28 images -> input dimension = 784
#--- hyperparameters ---
N_EPOCHS = 2
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_DEV = 100




#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'


class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=26, padding=1, kernel_size=5)
        self.conv2 = nn.Conv2d(26,20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,num_classes)
     

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args,model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
           
            


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss +=  F.nll_loss(output, target,reduction = "sum" ).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='Sign MNIST - Intro DL Homework 2')
    
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--alpha', type = float, default = 0,metavar = 'A',
                        help = 'regularisation parameter (default 0')
    parser.add_argument('--rtype', type = int, default = 0, metavar = 'L',
                        help = "Regularisation type L1/L2 (default none)")
    parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    
    
    
    # Training settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
    dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
    test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset = dev_set, batch_size = BATCH_SIZE_DEV, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)

    
   
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    


    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model,device,dev_loader, optimizer,loss_function, epoch)
        test(model, device, test_loader)


    
        torch.save(model.state_dict(), "sahin_cnn.pt")


if __name__ == '__main__':
    main()

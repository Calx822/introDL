import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#--- hyperparameters ---
N_EPOCHS = 20
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_DEV = 100




#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'



# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)




# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset = dev_set, batch_size = BATCH_SIZE_DEV, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)

#To fully specify a convolutional layer, we need input dim, filter size, number of filters, padding size and stride


INPUT_DIM = 784
n = 28
p = 1
f = 5

def psf_dim():
    return ((n + 2*p-f)/s) + 1

#--- model ---
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
        



#--- set up ---
# if torch.cuda.is_available():
device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

model = CNN().to(device)

# WRITE CODE HERE
optimizer = optim.Adam(params = model.parameters())
loss_function = nn.CrossEntropyLoss()

g = True

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(N_EPOCHS + 1)]

#--- training ---
for epoch in range(N_EPOCHS):
   
   
  for batch_num, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    l1_reg = 0.0
    alpha = 1e-3
    for param in model.parameters():
        l1_reg += alpha*torch.norm(param,1)
        loss = loss + l1_reg
    loss.backward()
    optimizer.step()
    if g == True:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_num * len(data), len(train_loader.dataset),
        100. * batch_num / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_num*100) + ((epoch-1)*len(train_loader.dataset)))
    #   torch.save(model.state_dict(), '/results/model.pth')
    #   torch.save(optimizer.state_dict(), '/results/optimizer.pth')
        

        # print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
        #       (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
        #        100. * train_correct / total, train_correct, total))
    
    # WRITE CODE HERE<
    # Please implement early stopping here.
    # You can try different versions, simplest way is to calculate the dev error and
    # compare this with the previous dev error, stopping if the error has grown.



#--- test ---

test_loss = 0
test_correct = 0
total = 0
correct = 0
model.eval()

with torch.no_grad():
    for data, target in test_loader:
      
      output = model(data)
      test_loss += loss_function(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
test_loss /= len(test_loader.dataset)
test_losses.append(test_loss)
print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
    test_loss, correct, len(test_loader.dataset), 
    100. * correct / len(test_loader.dataset)))



# with torch.no_grad():
#     for batch_num, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         test_loss += loss_function(output,target, size_average=False).item()
#         pred = output.data.max(1, keepdim=True)[1]
#         test_correct += pred.eq(target.data.view_as(pred)).sum()
        
#         test_loss /= len(test_loader.dataset)
#         test_loss.append(test_loss)
    
    
    
    
        # print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
        #       (batch_num, len(test_loader), test_loss / (batch_num + 1), 
        #        100. * test_correct / total, test_correct, total))


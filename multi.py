import torch
from torch import optim, nn, multiprocessing as mp
from torchvision import datasets, transforms
from torch.nn import functional as F
import copy
import time


mp.set_start_method("spawn", force=True)
DEV1 = "cuda:0"
DEV2 = "cuda:1"

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=True)


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # Dropout
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # Fully Connected Layer/Activation
        x = self.fc2(x)
        # Softmax gets probabilities.
        return F.log_softmax(x, dim=1)


def share_state_dict(state_dict):
    return {k: v.to("cpu").share_memory_() for k, v in state_dict.items()}


def train_epoch(epoch, model, optimizer, loader, global_params, queue, device):
    print("Starting process for", device)
    model.load_state_dict(global_params)
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx > 200:
            break
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
        loss = F.nll_loss(output, target)
        # dloss/dx for every Variable
        loss.backward()
        # to do a one-step update on our parameter.
        optimizer.step()
        # Print out the loss periodically.
        if batch_idx % 100 == 0:
            print('Train Epoch {}: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(device,
                                                                              epoch, batch_idx *
                                                                              len(data), len(
                                                                                  loader.dataset),
                                                                              100. * batch_idx / len(loader), loss.data.item()))
    queue.put(share_state_dict(model.state_dict()))
    time.sleep(10)


if __name__ == "__main__":
    model1 = Net()
    model2 = Net()

    global_params = model1.state_dict()
    queue = mp.Queue()

    model1.to(DEV1)
    model2.to(DEV2)

    optimizer1 = optim.SGD(model1.parameters(), 10e-3)
    optimizer2 = optim.SGD(model2.parameters(), 10e-3)

    p1 = mp.Process(target=train_epoch, args=(
        0, model1, optimizer1, train_loader, global_params, queue, DEV1))
    p2 = mp.Process(target=train_epoch, args=(
        0, model2, optimizer2, train_loader, model2.state_dict(), queue, DEV2))
    p1.start()
    p2.start()

    # p1.join()
    # p2.join()

    print("Processes done")
    for i in range(2):
        print(queue.get())

import torch
from torch import optim, nn, multiprocessing as mp
from torchvision import datasets, transforms
from torch.nn import functional as F
import copy
import time


DEV1 = "cuda:0"
DEV2 = "cuda:1"


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


def train_epoch(epoch, model, optimizer, loader, device):
    model.to(device)
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
    state_dict = model.cpu().state_dict()
    model.to(device)
    return state_dict


def worker(model, optimizer, loader, in_q, out_q, device):
    print("Starting worker for", device)
    while True:
        global_params = in_q.get()
        if global_params is None:
            print("Exiting worker for", device)
            break
        print("Loading global params for", device)
        model.load_state_dict(global_params)
        print("Training for", device)
        state_dict = train_epoch(
            0, model, optimizer, loader, device)
        print("Sending results for", device)
        out_q.put(share_state_dict(state_dict))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, num_workers=0)

    model1 = Net()
    model2 = Net()

    global_params = model1.cpu().state_dict()
    out_q = mp.Queue()
    in_q1 = mp.Queue()
    in_q2 = mp.Queue()

    model1.to(DEV1)
    model2.to(DEV2)

    optimizer1 = optim.SGD(model1.parameters(), 10e-3)
    optimizer2 = optim.SGD(model2.parameters(), 10e-3)

    p1 = mp.Process(target=worker, args=(
        model1, optimizer1, train_loader, in_q1, out_q, DEV1))
    p2 = mp.Process(target=worker, args=(
        model2, optimizer2, train_loader, in_q2, out_q, DEV2))
    p1.start()
    p2.start()

    in_q1.put(global_params)
    in_q2.put(global_params)
    in_q1.put(None)
    in_q2.put(None)

    # p1.join()
    # p2.join()

    print("Waiting for results...")
    for i in range(2):
        print(out_q.get())

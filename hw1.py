import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

diff = 1
num_epochs = 200
lr= 0.001

def get_num_weights(model):
    cnt = 0
    for w in model.parameters():
        l = list(w.size())
        if len(l)==2:
            cnt += list(w.size())[0]*list(w.size())[1]
        else:
            cnt += list(w.size())[0]
    return cnt

def torch_len(tensor):
    return list(tensor.size())[0]


def get_data():
    X1 = torch.randn(1000, 50)
    X2 = torch.randn(1000, 50) + diff
    X = torch.cat([X1, X2], dim=0)
    Y1 = torch.zeros(1000, 1)
    Y2 = torch.ones(1000, 1)
    Y = torch.cat([Y1, Y2], dim=0)


    p = torch.randperm(2000)
    X=X[p]
    Y=Y[p]

    X1 = torch.randn(50, 50)
    X2 = torch.randn(50, 50) + diff
    test_X = torch.cat([X1, X2], dim=0)
    Y1 = torch.zeros(50, 1)
    Y2 = torch.ones(50, 1)
    test_Y = torch.cat([Y1, Y2], dim=0)

    p = torch.randperm(100)
    test_X=test_X[p]
    test_Y=test_Y[p]

    print("X size: ", end="")
    print(X.size())
    print("Y size: ", end="")
    print(Y.size())

    print("X test size: ", end="")
    print(test_X.size())
    print("Y test size: ", end="")
    print(test_Y.size())

    return X,Y,test_X,test_Y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 100)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()


    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y





def train_epoch(model, opt, criterion, batch_size, X, Y):
    model.train()
    losses = []

    model.zero_grad()
    # (1) Forward
    y_out = net(X)
    # (2) Compute diff
    loss = criterion(y_out, Y)
    # (3) Compute gradients
    loss.backward()

    # (4) update weights
    for p in model.parameters():
        p.data -= p.grad.data * lr




net = Net()
criterion = nn.BCELoss()
X,Y,X_test,Y_test = get_data()

#train network
for e in range(num_epochs):
    train_epoch(model=net, opt=None, criterion=criterion, batch_size=50, X=X, Y=Y)


net.eval()
"Note: this notifies the network that it finished training. We don't actually need this line now," \
"since our network is primitive, but it is nice to have good habits for future works"

#run test set
out = net(X_test)
pred = torch.round(out).detach().numpy()

#convert ground truth to numpy
ynp = Y_test.data.numpy()

acc = np.count_nonzero(ynp==pred)

print("Number of Epochs: {}.".format(num_epochs))
print("Model accuracy: {}%".format(acc))
print(f'Number of Weights:{get_num_weights(net)}')

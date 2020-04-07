import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

diff = 1
num_epochs = 50
lr= 0.05
D_in,D_out = 50,1
D_H1,D_H2,D_H3 = 800,800,200

def Multiplot(l,xlabel,ylabel,title=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for p in l:
        x = p['x']
        y = p['y']
        funcName = p['funcName']
        plt.plot(x,y,label = funcName)
        plt.legend()
        plt.title(title)
        plt.plot()
    plt.show()

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

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_in, D_H1)
        self.fc2 = nn.Linear(D_H1, D_H2)
        self.fc3 = nn.Linear(D_H2,D_H3)
        self.out = nn.Linear(D_H3, D_out)
        self.out_act = nn.Sigmoid()
class NetTanh(BaseNet):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = torch.tanh(a1)
        #h1 = F.relu(a1)
        a2 = self.fc2(h1)
        #h2 = F.relu(a2)
        h2 = torch.tanh(a2)
        a3 = self.fc3(h2)
        #h3 = F.relu(a3)
        h3 = torch.tanh(a3)
        a4 = self.out(h3)
        y = self.out_act(a4)
        return y
class NetRelu(BaseNet):
    def __init__(self):
        super().__init__()    
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        a3 = self.fc3(h2)
        h3 = F.relu(a3)
        a4 = self.out(h3)
        y = self.out_act(a4)
        return y
class NetSigmoid(BaseNet):
    def __init__(self):
        super().__init__()        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = torch.sigmoid(a1)
        a2 = self.fc2(h1)
        h2 = torch.sigmoid(a2)
        a3 = self.fc3(h2)
        h3 = torch.sigmoid(a3)
        a4 = self.out(h3)
        y = self.out_act(a4)
        return y
def train_epoch(model, opt, criterion, batch_size, X, Y,x_test,y_test):
    model.train()
    #losses = []

    opt.zero_grad()
    # (1) Forward
    y_out = model(X)
    # (2) Compute diff
    loss = criterion(y_out, Y)
    # (3) Compute gradients
    loss.backward()

    # (4) update weights
    opt.step()
    detachedLoss = loss.detach()
    y_test_out = model(x_test)
    test_loss = criterion(y_test_out,y_test)
    test_loss_detached = test_loss.detach()
    return (detachedLoss.item(),test_loss_detached.item())
def trainModel(model,num_epochs,opt):
    trainLosses = []
    testLosses = []
    for e in range(num_epochs):
        curr_loss = train_epoch(model=model, opt=opt, criterion=criterion, batch_size=50,
        X=X,
        Y=Y,
        x_test=X_test,
        y_test=Y_test)
        trainLosses.append(curr_loss[0])
        testLosses.append(curr_loss[1])
    return [trainLosses,testLosses]
def testModel(model,x_test,y_test):
    out = model(x_test)
    pred = torch.round(out).detach().numpy()
    #convert ground truth to numpy
    ynp = y_test.data.numpy()
    acc = np.count_nonzero(ynp==pred)
    return acc

netTanh = NetTanh()
criterion = nn.BCELoss()
X,Y,X_test,Y_test = get_data()

tanhTestLosses = []
tanhTrainLosses = []
#train network
optimizer = torch.optim.Adam(params=netTanh.parameters(),lr=lr)
testTrainlosses = trainModel(model=netTanh,num_epochs=num_epochs,opt=optimizer)
netTanh.eval()

x = np.arange(1,len(testTrainlosses[0])+1)
dtrainLosses = {'x':x,'y':testTrainlosses[0],'funcName':'Train losses'}
dtestLosses = {'x':x,'y':testTrainlosses[1],'funcName':'Test losses'}
Multiplot([dtrainLosses,dtestLosses],'#Epochs','Accuracy',title='Tanh')

acc = testModel(netTanh,X_test,Y_test)
print("Number of Epochs: {}.".format(num_epochs))
print("Model accuracy: {}%".format(acc))
print(f'Number of Weights:{get_num_weights(netTanh)}')

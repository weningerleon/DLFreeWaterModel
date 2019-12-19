import numpy as np
import torch
import torch.nn as nn

device = torch.device('cpu')


class DifreewaterNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DifreewaterNet, self).__init__()
        # Calling Super Class's constructor
        s = input_dim
        self.lin1 = nn.Linear(s, np.int(s/2))
        self.lin2 = nn.Linear(np.int(s/2), np.int(s/4))
        self.lin4 = nn.Linear(np.int(s/4), np.int(s/8))
        self.lin8 = nn.Linear(np.int(s/8), output_dim)
        self.act_fn = nn.Tanh()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act_fn(x)
        x = self.lin2(x)
        x = self.act_fn(x)
        x = self.lin4(x)
        x = self.act_fn(x)
        return self.lin8(x)


def train(x, y, net, criterion, num_epochs, optimizer):
    len_data = x.__len__()
    s = int(len_data*0.8)

    x_train = x[:s,:]
    y_train = y[:s]
    x_val = x[s:,:]
    y_val = y[s:]

    for j in range(num_epochs):
        train_epoch(x_train, y_train, net, criterion, optimizer, batch_size=256)

        if j%10==0:
            mse_val, me_val = evaluate(x_val, y_val, net)
            print("Epoch {}: MSE={}, ME={}".format(j,mse_val, me_val))

    return mse_val, me_val


def train_epoch(x_train, y_train, net, criterion, optimizer, batch_size=256):
    for i in range(0, x_train.__len__(), batch_size):
        input = x_train[i:i+batch_size]
        target = np.expand_dims(y_train[i:i+batch_size], axis=-1)

        input = np.float32(input)
        target = np.float32(target)

        input = torch.from_numpy(input).to(device)
        target = torch.from_numpy(target).to(device)
        net = net.to(device)

        optimizer.zero_grad()
        output = net.forward(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(x_val, y_val, net):

    output = predict(x_val, net)

    target = np.expand_dims(y_val, axis=-1)

    mse = np.mean((output - target)**2)
    me = np.mean(np.abs(output - target))

    return mse, me



def predict(data, net):

    input = torch.from_numpy(np.float32(data)).to(device)
    output = net(input)

    prediction = output.cpu().detach().numpy()

    prediction[prediction>1]=1
    prediction[prediction<0]=0


    return prediction

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train(x, y, num_epochs=25):
    net = DifreewaterNet(x.shape[1], 1)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    len_data = x.__len__()
    s = int(len_data*0.8)

    x_train = x[:s,:]
    y_train = y[:s]
    x_val = x[s:,:]
    y_val = y[s:]
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(np.expand_dims(y_train, axis=-1)).float()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(np.expand_dims(y_val, axis=-1)).float()

    trainset = TensorDataset(x_train_t, y_train_t)
    testset = TensorDataset(x_val_t, y_val_t)

    trainloader = DataLoader(trainset, batch_size=256, num_workers=2, shuffle=True)
    testloader = DataLoader(testset, batch_size=2560, num_workers=2, shuffle=True)

    for epoch in range(num_epochs):
        for batch_nr, data in enumerate(trainloader):
            input, tgt = data
            input = input.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            output = net.forward(input)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
        # Validate
        with torch.no_grad():
            mse = []
            me = []
            for batch_nr, data in enumerate(testloader):
                input, tgt = data
                input = input.to(device)
                output = net(input)
                output = output.detach().cpu()

                mse.append(np.mean((output - tgt).numpy() ** 2))
                me.append(np.mean(np.abs((output - tgt).numpy())))
            print("Epoch {}: MSE={}, ME={}".format(epoch, np.mean(mse), np.mean(me)))

    return net


def predict(data, net):
    with torch.no_grad():
        input = torch.from_numpy(np.float32(data)).to(device)
        output = net(input)

        prediction = output.cpu().numpy()

    prediction[prediction>1]=1
    prediction[prediction<0]=0

    return prediction

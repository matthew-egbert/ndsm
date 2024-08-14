from pylab import *
import itertools
import os
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        N = 4
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, 2),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__' :
    N = 10000    
    N_BATCHES = 500
    BATCH_SIZE = N // N_BATCHES
    print(f'BATCH_SIZE: {BATCH_SIZE}')

    x = np.random.choice([0,1],p=[0.5,0.5],size=(N,2)) ## input
    y = np.apply_along_axis(lambda x : x[0] ^ x[1],1,x) ## classifying output (COL1 is AND, COL2 is NAND)
    y2 = np.array(1.0 - y)
    y = np.vstack([y,y2]).T

    x = x.reshape(-1,2)
    y = y.reshape(-1,2)
    # for i in range(N) :
    #     print(f'{x[i]} -> {y[i]}')
    # quit()

    device = "cpu" ; print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()
    learning_rate = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    model.train()
    losses = []
    for batch_i in range(0,N_BATCHES) :
        start_i = batch_i*BATCH_SIZE
        stop_i = batch_i*BATCH_SIZE+BATCH_SIZE
        X = torch.tensor(x[start_i:stop_i],dtype=torch.float32).to(device)
        Y = torch.tensor(y[start_i:stop_i],dtype=torch.float32).to(device)

        model_out = model(X)
        loss = loss_fn(model_out,Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    plot(losses)
    


    for test_input in [[0,0],
                       [0,1],
                       [1,0],
                       [1,1],
                       ] :

        X = torch.tensor(test_input,dtype=torch.float32).to(device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=0)(logits)
        y_pred = pred_probab.argmax(0)
        print(f"Prediction class: {test_input}  --> {y_pred} :: {logits}")

    show()



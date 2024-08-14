from pylab import *
import itertools
import os
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        NeuralNetwork model for learning a probabalistic relationship between a
        sensorimotor history and 
        """

        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Brain(object) :
    def __init__(self, model, *args, **kwargs) :
        self.model = model
        body : Body = self.model.body
        self.num_distinct_motor_states = body.N_ALLOWED_MOTOR_VALUES ** 2
        self.nn = NeuralNetwork()

if __name__ == '__main__' :
    N_DATA = 10000    
    N_BATCHES = 1000
    BATCH_SIZE = 110
    LEARNING_RATE = 1e-2
    P_NOT_IDENTITY = 0.1
    print(f'N_DATA: {N_DATA}')
    print(f'N_BATCHES: {N_BATCHES}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'LEARNING_RATE: {LEARNING_RATE}')

    x = np.random.choice([0,1],p=[0.5,0.5],size=(N_DATA,1)) ## input
    y = np.array(x)
    for i in range(1,len(y)) :
        if np.random.rand() < P_NOT_IDENTITY :
            y[i] = 1.0-y[i]
    y2 = np.array(1.0-y)
    y = np.concatenate((y,y2),axis=1)

    x = x.reshape(-1,1)
    y = y.reshape(-1,2)

    device = "cpu" ; print(f"Using {device} device")

    model = NeuralNetwork(1,16,2).to(device)
    #print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.train()
    losses = []
    for batch_i in range(0,N_BATCHES) :
        start_i = batch_i*BATCH_SIZE
        x_data = np.roll(x,-start_i,axis=0)[0:BATCH_SIZE]
        y_data = np.roll(y,-start_i,axis=0)[0:BATCH_SIZE]
        X = torch.tensor(x_data,dtype=torch.float32).to(device)
        Y = torch.tensor(y_data,dtype=torch.float32).to(device)

        model_out = model(X)
        loss = loss_fn(model_out,Y)
        # for i in range(0,len(X)) :
        #     print(f"X: {X[i]} Y: {Y[i]} model_out: {model_out[i]} loss: {loss}") 

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    plot(losses)    

    for test_input in [[0],
                       [0],
                       [1],
                       [1],
                       ] :

        X = torch.tensor(test_input,dtype=torch.float32).to(device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=0)(logits)
        y_pred = pred_probab.argmax(0)
        print(f"Prediction class: {test_input}  --> {y_pred} :: {logits}  >>> {pred_probab}")

    show()



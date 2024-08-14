from pylab import *
import itertools
import os
import torch
from torch import nn


class Brain(object) :
    def __init__(self, model, *args, **kwargs) :
        pass

    def generate_training_data(self,N_DATA=1000) :
        history_length = 3
        sensor_state = [0,1]
        n_sensors = 2
        n_motors = 2

        all_sensor_states = list(itertools.product(sensor_state,repeat=n_sensors))
        self.all_possible_histories = list(itertools.product(all_sensor_states,repeat=history_length))

        def sh_to_str(sh) :
            return str(list(flatten(sh)))
        # plms = linspace(0,1,len(self.all_possible_histories))
        # plms /= sum(plm)

        ## create some probabilities for likelihood of each sensory history
        plms = [sum(h) for h in self.all_possible_histories]
        plms /= max(plms)
        prms = max(plms) - plms
        prms /= max(prms)

        plms = np.array(plms)
        plms *= 0
        indices = []
        for sh in self.all_possible_histories :
            if sh[-1] == (0,0) :
                indices.append(True)
            else :
                indices.append(False)
        plms[indices] = 1.0
        prms = plms

        lm_probs = {}
        rm_probs = {}
        for h,plm,prm in zip(self.all_possible_histories,plms,plms) :
            lm_probs[sh_to_str(h)] = plm
            rm_probs[sh_to_str(h)] = prm
            # print(f'{sh_to_str(h)} : {plm:.3f}, {prm:.3f}')
        figure(figsize=(12,4))
        step(range(len(self.all_possible_histories)),plms,where='mid',label='p(lm|sensor history)')
        step(range(len(self.all_possible_histories)),prms,where='mid',label='p(rm|sensor history)')
        gca().set_xticks(range(len(self.all_possible_histories)))
        gca().set_xticklabels([str(list(h)) for h in self.all_possible_histories],rotation=90)
        title('Likelihood of motor values for sensor history')
        legend()
        tight_layout()
        savefig('p(m|s_h).png')
        close()

        """ randomly generate a sensory history """
        s_h = []
        for _ in range(N_DATA):
            s_index = np.random.choice(range(len(all_sensor_states)))
            s_h.append(all_sensor_states[s_index])

        """ randomly generate a history of motor values that is consistent with the sensory history """
        lm_h = [0,0]
        rm_h = [0,0]
        for t in range(3,N_DATA+1):
            plm = lm_probs[sh_to_str(s_h[t-3:t])]
            lm = np.random.choice([0,1],p=[1.0-plm,plm])
            lm_h.append(lm)

            prm = rm_probs[sh_to_str(s_h[t-3:t])]
            rm = np.random.choice([0,1],p=[1.0-prm,prm])
            rm_h.append(rm)
            #print(f'{sh_to_str(s_h[t-3:t])} : {plm,prm} --> {lm,rm}')

        lm_h = np.array(lm_h).reshape(-1,1)
        rm_h = np.array(rm_h).reshape(-1,1)

        return np.hstack([s_h,lm_h,rm_h])

def data_to_hist(data) :
    """ generate a histogram that shows how often the last two columns show up given the first 6 columns """
    hist = {}
    for t in range(3,len(data)) :
        s_h = str(np.concatenate([data[t-3,:2],data[t-2,:2],data[t-1,:2]]))
        m = str(data[t-1,2:])
        #(f'{t}] {s_h} : {m}')
        if s_h not in hist :
            hist[s_h] = {'[0 0]' : 0, '[0 1]' : 0, '[1 0]' : 0, '[1 1]' : 0}
        hist[s_h][m] += 1

    ks = sorted(hist.keys())
    histogram_data = np.zeros((4,len(ks)))
    for i,sh in enumerate(ks) :
        m = hist[sh]
        total_count = m['[0 0]'] + m['[0 1]'] + m['[1 0]'] + m['[1 1]']
        histogram_data[0,i] = m['[0 0]'] / total_count
        histogram_data[1,i] = m['[0 1]'] / total_count
        histogram_data[2,i] = m['[1 0]'] / total_count
        histogram_data[3,i] = m['[1 1]'] / total_count
    figure(figsize=(12,4))
    imshow(histogram_data)
    gca().set_xticks(range(len(ks)))
    gca().set_yticks(range(4))
    gca().set_xticklabels(list(ks),rotation=90)
    gca().set_yticklabels(['[0 0]','[0 1]','[1 0]','[1 1]'])        
    tight_layout()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*2, 3*2),
            nn.ReLU(),
            nn.Linear(3*2, 3*2),
            nn.ReLU(),
            nn.Linear(3*2, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__' :
    b = Brain(None)
    training_data = b.generate_training_data(10000)
    
    #data_to_hist(training_data)
    #savefig('training_data_distribution.png')

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    device = "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    data = training_data
    losses = []
    for t in range(3,len(data)) :
        s_h = np.concatenate([data[t-3,:2],data[t-2,:2],data[t-1,:2]])
        s_h = [s_h,]
        actual_m = torch.tensor([data[t-1,2:],],dtype=torch.float32).to(device)

        X = torch.tensor(s_h,dtype=torch.float32).to(device)
        predicted_motor = model(X)
        loss = loss_fn(predicted_motor,actual_m)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())            


    for test_input in [[0,0,0,0,0,0],
                      [0,0,1,1,1,1],
                      [0,0,1,1,1,1],
                      [0,0,0,0,1,1],
                      [0,0,1,1,1,1],
                      [1,1,0,0,0,0],
                      [1,1,1,1,1,1],] :

        X = torch.tensor([test_input,],dtype=torch.float32).to(device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        print(f"Prediction class: {test_input} --> {y_pred} :: {logits}")

    plot(losses,',')
    show()


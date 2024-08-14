from pylab import *
import itertools
import os
import torch
from torch import nn
from body import Body

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        NeuralNetwork model for learning a probabalistic relationship between a
        sensorimotor history and
        """

        super().__init__()
        #self.flatten = nn.Flatten()
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
    def __init__(self, model, sm_duration=3, *args, **kwargs) :
        """
        Brain object for the model. The sm_duration parameter specifies the
        number of time steps of sm history the brain considers.
        """
        self.model = model
        self.sm_duration = sm_duration
        self.body : Body = self.model.body

        N_SENSORS = self.body.n_sensors
        N_MOTORS  = self.body.n_motors
        nn_input_size = (N_SENSORS + N_MOTORS) * self.sm_duration
        nn_hidden_size = 20
        nn_output_size = (self.body.N_ALLOWED_SENSOR_VALUES ** N_SENSORS) * (self.body.N_ALLOWED_MOTOR_VALUES ** N_MOTORS)

        self.nn = NeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size)
        self.device = "cpu" ; print(f"Using {self.device} device")
        self.n_model = self.nn.to(self.device)
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.SGD(self.n_model.parameters(), lr=self.learning_rate)
        self.prediction_error = 0.0
        self.prediction_errors = np.zeros(self.model.TIMESERIES_LENGTH)
        self.prediction_h = np.zeros((N_SENSORS+N_MOTORS, self.model.TIMESERIES_LENGTH))

        self.recent_sms_h = np.zeros((N_SENSORS+N_MOTORS,self.sm_duration))
        self.output_probabilities = np.zeros(nn_output_size)

    def update_learning_variables(self) :
        """ Updates the things that the network learns from, i.e. the recent
        sensorimotor history and the recent motor output. """
        input_final_it = self.model.it-2
        correct_output_it = self.model.it-1
        self.recent_sms_h = np.roll(self.body.sms_h,-(input_final_it-1),axis=1)[:,0:self.sm_duration]
        
        self.correct_sms = np.roll(self.body.sms_h,-(correct_output_it),axis=1)[:,0].flatten()
        self.correct_sms = tuple(self.correct_sms)
        self.correct_output = self.body.smcodec.to_onehot(self.correct_sms)
        
    def learn(self) :
        loss_fn = nn.CrossEntropyLoss()
        #loss_fn = nn.MSELoss()
        self.n_model.train()

        self.update_learning_variables()
        nn_input = torch.tensor(self.recent_sms_h.flatten(),dtype=torch.float32).to(self.device)
        correct_nn_output = torch.tensor(self.correct_output,dtype=torch.float32).to(self.device)

        model_out = self.n_model(nn_input)
        loss = loss_fn(model_out,correct_nn_output)
            
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.prediction_errors[self.model.it%self.model.TIMESERIES_LENGTH] = loss.item()
        self.prediction_error = loss.item()
        #ps = model_out.detach().numpy()
        ps = nn.Softmax(dim=0)(model_out)
        ps = ps.detach().numpy()
        self.output_probabilities = ps

        output_state_index = np.random.choice(range(len(ps)), p=ps)
        output_state = self.body.smcodec.index_to_values(output_state_index)
        output_state = np.array(list(output_state))
        
        # print()
        # print(f'nn_input: {nn_input}')
        # print(f'correct_nn_output: {correct_nn_output}')# => lm: {olm} rm: {orm}')
        # print(f'model_out: {model_out} loss: {loss}')        
        # print(f'ps: {ps} argmax: {argmax(ps)}')
        # print(f'output_state_index: {output_state_index}')
        # print(f'output_state: {output_state}')
        # print(f'correct_sms: {[float(x) for x in self.correct_sms]}')
        self.prediction_h[:,self.model.it%self.model.TIMESERIES_LENGTH] = output_state


    def prepare_to_iterate(self) :
        self.learn()

    def iterate(self) :
        pass
        #self.learn()


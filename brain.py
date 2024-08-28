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
    def __init__(self, model, sm_duration=100, *args, **kwargs) :
        """
        Brain object for the model. The sm_duration parameter specifies the
        number of time steps of sm history the brain considers.
        """
        self.model = model
        self.sm_duration = sm_duration
        self.body : Body = self.model.body
        self.learning_rate_exponent = -3
        self.DETERMINISTIC_NN_OUTPUT = False
        self.ZERO_LEARNING_RATE = False

        self.N_SENSORS = len(self.body.sensors)
        self.N_MOTORS  = len(self.body.motors)
        nn_input_size = (self.N_SENSORS + self.N_MOTORS) * self.sm_duration
        nn_output_size = len(self.body.onehotter.onehot)
        nn_hidden_size = nn_output_size*2

        my_nn = NeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size)
        print(my_nn)

        self.device = "cuda" ; print(f"Using {self.device} device")
        self.n_model = my_nn.to(self.device)
        self.learning_rate = exp(self.learning_rate_exponent)
        self.optimizer = torch.optim.SGD(self.n_model.parameters(), lr=self.learning_rate)
        self.prediction_error = 0.0
        self.prediction_errors = np.zeros(self.model.TIMESERIES_LENGTH)
        self.prediction_h = np.zeros((self.N_SENSORS+self.N_MOTORS, self.model.TIMESERIES_LENGTH))

        self.recent_sms_h = np.zeros((self.N_SENSORS+self.N_MOTORS,self.sm_duration))
        self.debug_indices = np.zeros_like(self.body.sms_h)
        self.output_probabilities = np.zeros(nn_output_size)
        self.output_probabilities_h = np.zeros((nn_output_size,self.model.TIMESERIES_LENGTH))
        self.most_recent_output = np.zeros(nn_output_size)

    def get_input_output_columns(self, τ : int=0, mod=None) :
        """
        τ : the index of the data that is the target output

        mod : if None, this defaults to the length of the SMS_h buffer
              if not None, it specifies the length of the buffer to use 
        
        returns a tuple (input, correct_output) where         

            input is the sensorimotor history of length sm_duration
            correct_output is the sensorimotor state at iteration its_ago            
        """        

        if mod is None :
            mod = np.shape(self.body.sms_h)[1]

        input_cols = arange(τ-self.sm_duration,τ) % mod
        output_col = (τ) % mod

        return input_cols, output_col

    def get_learning_pair(self, τ : int=1, sms_array = None) :
        """ 
        τ : the index of the data that is the target output
        sms_array : the sensorimotor history to use for training (defaults to self.body.sms_h)
        """
        
        if sms_array is None :
            sms_array = self.body.sms_h
            learning_input_cols,target_output_col = self.get_input_output_columns(τ=τ)
        else :
            ## used when training from a file
            learning_input_cols,target_output_col = self.get_input_output_columns(τ=τ,mod=len(sms_array))

        # The sensorimotor history just before the target output is the NN's INPUT
        input = sms_array[:,learning_input_cols].flatten()
        # The SMS that follows it is the NN's TARGET OUTPUT
        self.body.onehotter.values = sms_array[:,target_output_col]#.flatten()
        ## but we need the correct output as a onehot
        correct_output = self.body.onehotter.onehot

        return input, correct_output

    def train_on_file(self, filename) :
        self.optimizer.zero_grad()
        self.learning_rate = 10.0**(self.learning_rate_exponent) * (not self.ZERO_LEARNING_RATE)
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

        loss_fn = nn.CrossEntropyLoss()
        self.n_model.train()

        training_data = np.load(filename)
        training_data = training_data.T

        inputs = []
        outputs = []
        for τ in [self.sm_duration+1,np.shape(training_data)[1]] :
            i,o = self.get_learning_pair(τ=τ,sms_array=training_data) ## WRONG
            inputs.append(i)
            outputs.append(o)
        
        nn_input = torch.tensor(inputs,dtype=torch.float32).to(self.device)
        correct_nn_output = torch.tensor(outputs,dtype=torch.float32).to(self.device)

        model_out = self.n_model(nn_input)
        loss = loss_fn(model_out,correct_nn_output)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        self.prediction_error = loss.item()        

    def learn(self) :
        self.optimizer.zero_grad()
        self.learning_rate = 10.0**(self.learning_rate_exponent) * (not self.ZERO_LEARNING_RATE)
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

        loss_fn = nn.CrossEntropyLoss()
        self.n_model.train()

        inputs = []
        outputs = []
        for τ in [1,128] :
            i,o = self.get_learning_pair(τ=self.model.it-τ)
            inputs.append(i)
            outputs.append(o)
        
        nn_input = torch.tensor(inputs,dtype=torch.float32).to(self.device)
        correct_nn_output = torch.tensor(outputs,dtype=torch.float32).to(self.device)

        model_out = self.n_model(nn_input)
        loss = loss_fn(model_out,correct_nn_output)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        self.prediction_error = loss.item()

    def act(self) :
        self.n_model.eval()
        
        action_input_cols, _ = self.get_input_output_columns(τ=0)
        action_input = self.body.sms_h[:,action_input_cols]
        nn_input = torch.tensor(action_input.flatten(),dtype=torch.float32).to(self.device)
        model_out = self.n_model(nn_input)
        ps = nn.Softmax(dim=0)(model_out)
        ps = ps.cpu().detach().numpy()
        self.output_probabilities = ps

        if self.DETERMINISTIC_NN_OUTPUT :
            output_state_index = argmax(ps)
        else :
            output_state_index = np.random.choice(range(len(ps)), p=ps)

        output_onehot = np.zeros(len(ps))
        output_onehot[output_state_index] = 1

        self.body.onehotter.onehot = output_onehot
        output_values = self.body.onehotter.values        
        output_indices = self.body.onehotter.indices
        
        self.actual_nn_output = output_values

        ## update visualizations
        self.most_recent_output[:] *= 0
        self.most_recent_output[output_state_index] = 1
        self.prediction_errors[self.model.it%self.model.TIMESERIES_LENGTH] = self.prediction_error
        self.prediction_h[:,self.model.it%self.model.TIMESERIES_LENGTH] = output_indices

        x = np.log(self.output_probabilities,where=self.output_probabilities>0)
        x -= min(x)
        x /= max(x)
        self.output_probabilities_h[:,self.model.it%self.model.TIMESERIES_LENGTH] = x

    def prepare_to_iterate(self) :
        self.learn()
        self.act()

    def iterate(self) :
        pass



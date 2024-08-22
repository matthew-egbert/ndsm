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
        self.learning_rate_exponent = -3
        self.DETERMINISTIC_NN_OUTPUT = False
        self.ZERO_LEARNING_RATE = False

        self.N_SENSORS = self.body.n_sensors
        self.N_MOTORS  = self.body.n_motors        
        nn_input_size = (self.N_SENSORS + self.N_MOTORS) * self.sm_duration        
        nn_output_size = (self.body.N_ALLOWED_SENSOR_VALUES ** self.N_SENSORS) * (self.body.N_ALLOWED_MOTOR_VALUES ** self.N_MOTORS)
        nn_hidden_size = nn_output_size*2

        my_nn = NeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size)        
        self.device = "cpu" ; print(f"Using {self.device} device")
        self.n_model = my_nn.to(self.device)
        self.learning_rate = exp(self.learning_rate_exponent)
        self.optimizer = torch.optim.SGD(self.n_model.parameters(), lr=self.learning_rate)
        self.prediction_error = 0.0
        self.prediction_errors = np.zeros(self.model.TIMESERIES_LENGTH)
        self.prediction_h = np.zeros((self.N_SENSORS+self.N_MOTORS, self.model.TIMESERIES_LENGTH))
    
        self.recent_sms_h = np.zeros((self.N_SENSORS+self.N_MOTORS,self.sm_duration))
        self.debug_indices = np.zeros_like(self.body.sms_h)
        self.output_probabilities = np.zeros(nn_output_size)
        self.most_recent_output = np.zeros(nn_output_size)

    def update_learning_variables(self) :
        """ Updates the things that the network learns from, i.e. the recent
        sensorimotor history and the recent motor output. """

        input_cols = arange(self.model.it-(self.sm_duration+1),self.model.it-1) % np.shape(self.body.sms_h)[1]
        target_output_col = (self.model.it-1) % np.shape(self.body.sms_h)[1]
        #print(f'LEARNING: INPUTCOLS: {input_cols.tolist()} => OUTPUTCOL: {target_output_col}')

        # The recent sensorimotor history is the NN's INPUT
        self.recent_sms_h = self.body.sms_h[:,input_cols]            
        

        # The SMS just after that recent sensorimotor history is the NN's TARGET OUTPUT
        self.correct_sms = self.body.sms_h[:,target_output_col].flatten()
        self.correct_sms = tuple(self.correct_sms)
        self.correct_output = self.body.smcodec.to_onehot(self.correct_sms)

        #print(self.correct_output)
        # if self.model.it > 65 :
        #     print(self.recent_sms_h)
        #     print(self.correct_sms)
        #     quit()
                    
    def learn(self) :
        self.optimizer.zero_grad()
        self.learning_rate = 10.0**(self.learning_rate_exponent) * (not self.ZERO_LEARNING_RATE)
        for g in self.optimizer.param_groups:
            #print(self.learning_rate)
            g['lr'] = self.learning_rate 

        loss_fn = nn.CrossEntropyLoss() #loss_fn = nn.MSELoss()        
        self.n_model.train()

        self.update_learning_variables()
        nn_input = torch.tensor(self.recent_sms_h.flatten(),dtype=torch.float32).to(self.device)
        correct_nn_output = torch.tensor(self.correct_output,dtype=torch.float32).to(self.device)

        model_out = self.n_model(nn_input)
        loss = loss_fn(model_out,correct_nn_output)
                
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        

        self.prediction_error = loss.item()
                        
        ps = nn.Softmax(dim=0)(model_out)
        ps = ps.detach().numpy()
        self.output_probabilities = ps

        output_state_index = np.random.choice(range(len(ps)), p=ps)
        if self.DETERMINISTIC_NN_OUTPUT :
            output_state_index = argmax(ps)

        output_state = self.body.smcodec.index_to_values(output_state_index)
        output_state = np.array(list(output_state))

        self.next_motor_state = output_state[self.N_SENSORS:]

        ## update visualizations
        self.most_recent_output[:] *= 0
        self.most_recent_output[output_state_index] = 1        
        self.prediction_errors[self.model.it%self.model.TIMESERIES_LENGTH] = self.prediction_error
        self.prediction_h[:,self.model.it%self.model.TIMESERIES_LENGTH] = output_state
        
        

    def prepare_to_iterate(self) :
        self.learn()

    def iterate(self) :
        pass
        #self.learn()


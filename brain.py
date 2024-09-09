from pylab import *
import itertools
import os
import torch
from torch import nn
from body import Body
from discval import OneHotter
from plotting_utils import better_colorbar

# params = {
#     'image.origin': 'lower',
#     'image.interpolation': 'nearest',
#     'image.cmap': 'gray',
#     'axes.grid': False,
#     'savefig.dpi': 150,  # to adjust notebook inline plot size
#     'axes.labelsize': 8, # fontsize for x and y labels (was 10)
#     'axes.titlesize': 8,
#     'font.size': 8, # was 10
#     'legend.fontsize': 6, # was 10
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'text.usetex': True,
#     'figure.figsize': [3.39, 2.10],
#     'font.family': 'serif',
# }
# matplotlib.rcParams.update(params)

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

class NDSMDataSet(torch.utils.data.Dataset):
    def __init__(self, sms_h : np.ndarray, onehotter : OneHotter, input_duration : int) :        
        self.data = sms_h        
        self.input_duration = input_duration
        self.onehotter = onehotter

    def __len__(self) :
        return np.shape(self.data)[1]

    def __getitem__(self, idx) :
        input_cols = arange(idx-self.input_duration,idx) % np.shape(self.data)[1]
        output_col = (idx) % np.shape(self.data)[1]
        
        output_sms = self.onehotter.values = self.data[:,output_col]
        output_onehot = self.onehotter.onehot
        
        return self.data[:,input_cols].flatten(), output_onehot

class Brain(object) :
    def __init__(self, model, input_duration=100, *args, **kwargs) :
        """
        Brain object for the model. The sm_duration parameter specifies the
        number of time steps of sm history the brain considers.
        """
        self.model = model
        self.input_duration = input_duration
        self.body : Body = self.model.body
        self.learning_rate_exponent = -3
        self.DETERMINISTIC_NN_OUTPUT = False
        self.ZERO_LEARNING_RATE = False

        self.N_SENSORS = len(self.body.sensors)
        self.N_MOTORS  = len(self.body.motors)
        nn_input_size = (self.N_SENSORS + self.N_MOTORS) * self.input_duration
        nn_output_size = len(self.body.onehotter.onehot)
        nn_hidden_size = nn_input_size * 2 // 3 + nn_output_size

        my_nn = NeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size)
        print(my_nn)

        self.device = "cuda" ; print(f"Using {self.device} device")
        self.n_model = my_nn.to(self.device)
        self.learning_rate = exp(self.learning_rate_exponent)
        self.optimizer = torch.optim.SGD(self.n_model.parameters(), lr=self.learning_rate, momentum=0.0)
        self.prediction_error = 0.0
        self.prediction_errors = np.zeros(self.model.TIMESERIES_LENGTH)
        self.prediction_h = np.zeros((self.N_SENSORS+self.N_MOTORS, self.model.TIMESERIES_LENGTH))

        self.recent_sms_h = np.zeros((self.N_SENSORS+self.N_MOTORS,self.input_duration))
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

        input_cols = arange(τ-self.input_duration,τ) % mod
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

        print(f'Loading training data from {filename}')
        print(f'size: {np.load(filename).shape}')
        training_data = NDSMDataSet(np.load(filename).T,self.body.onehotter,input_duration=self.input_duration)
        training_data_loader = torch.utils.data.DataLoader(training_data,batch_size=2,shuffle=False,pin_memory=True) ## setting shuffle to True may be dissimilar to usual training

        running_loss = 0
        for i,data in enumerate(training_data_loader) :
            inputs, outputs = data
            
            self.optimizer.zero_grad()

            nn_input = torch.tensor(inputs,dtype=torch.float32).to(self.device)
            correct_nn_output = torch.tensor(outputs,dtype=torch.float32).to(self.device)
            # print(f'correct_nn_output: {correct_nn_output}')

            model_out = self.n_model(nn_input)
            # print(f'model_out: {model_out}')
            loss = loss_fn(model_out,correct_nn_output)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        #self.prediction_error = loss.item

    def learn(self) :
        self.n_model.train()
        self.optimizer.zero_grad()
        self.learning_rate = 10.0**(self.learning_rate_exponent) * (not self.ZERO_LEARNING_RATE)
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

        loss_fn = nn.CrossEntropyLoss()

        inputs = []
        outputs = []
        for τ in [1]: #range(1,64) :
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
        
        action_input_cols, _ = self.get_input_output_columns(τ=self.model.it)
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
        #output_indices = self.body.onehotter.indices
        
        self.actual_nn_output = output_values

        ## update visualizations
        self.most_recent_output[:] *= 0
        self.most_recent_output[output_state_index] = 1
        self.prediction_errors[self.model.it%self.model.TIMESERIES_LENGTH] = self.prediction_error
        self.prediction_h[:,self.model.it%self.model.TIMESERIES_LENGTH] = output_values

        x = np.log(self.output_probabilities,where=self.output_probabilities>0)
        x -= min(x)
        x /= max(x)
        self.output_probabilities_h[:,self.model.it%self.model.TIMESERIES_LENGTH] = x

    def prepare_to_iterate(self) :
        self.learn()
        self.act()

    def image_2d_output(self) :
        self.n_model.eval()
        action_input_cols, _ = self.get_input_output_columns(τ=self.model.it)
        
        ## this is the base input to the NN
        base_input = self.body.sms_h[:,action_input_cols]
        ## we modify the last col systematically, to show what the influence
        ## of the last state is upon the output of the NN

        def sample_output(x) :
            base_input[0,-1] = x[1]
            base_input[1,-1] = x[0]
            nn_input = torch.tensor(base_input.flatten(),dtype=torch.float32).to(self.device)
            model_out = self.n_model(nn_input)
            ps = nn.Softmax(dim=0)(model_out)
            ps = ps.cpu().detach().numpy()

            # DETERMINISTIC_NN_OUTPUT
            output_state_index = argmax(ps)                
            output_onehot = np.zeros(len(ps))
            output_onehot[output_state_index] = 1

            self.body.onehotter.onehot = output_onehot
            output_values = self.body.onehotter.values        
            #output_indices = self.body.onehotter.indices

            return output_values[1],output_values[0]

        mesh = np.meshgrid(self.body.motors[0].allowed_values,self.body.sensors[0].allowed_values)
        outputs = np.apply_along_axis(sample_output,0,mesh)

        ## plot a vector field
        figure(figsize=(14,5.5))
        m = outputs[0]
        s = outputs[1]

        pxlw = (0.75 - -0.75)/np.shape(m)[1]/2
        pxlh = (1.0 - 0.0)/np.shape(m)[0]/2
        extnts = (-0.75-pxlw,0.75+pxlw,0-pxlh,1+pxlh)
        subplot2grid((1,2),(0,0))
        img = imshow(m,origin='lower',extent=extnts,vmin=-0.75,vmax=0.75,cmap='RdGy')
        #better_colorbar(img)
        #quiver(mesh[0],mesh[1],outputs[0]-mesh[0],outputs[1]-mesh[1], pivot='mid')
        for i in range(np.shape(m)[0]):
            for j in range(np.shape(m)[1]):
                plot([mesh[0][i][j], outputs[0][i][j]], 
                     [mesh[1][i][j], outputs[1][i][j]], color='blue', alpha=0.3, lw=0.5)
        xlim(extnts[0],extnts[1])
        ylim(extnts[2],extnts[3])
        xlabel('motor output')

        subplot2grid((1,2),(0,1))
        img = imshow(s,origin='lower',extent=extnts,vmin=0.0,vmax=1.0,cmap='Purples_r')
        #better_colorbar(img)
        #quiver(mesh[0],mesh[1],outputs[0]-mesh[0],outputs[1]-mesh[1], pivot='mid')
        for i in range(np.shape(m)[0]):
            for j in range(np.shape(m)[1]):
                plot([mesh[0][i][j], outputs[0][i][j]], 
                     [mesh[1][i][j], outputs[1][i][j]], color='black', alpha=0.3, lw=0.5)
        xlim(extnts[0],extnts[1])
        ylim(extnts[2],extnts[3])
        xlabel('sensor output')
        
        #suptitle(f't = {self.model.it*self.model.DT}')
        tight_layout()
        savefig(f'nn_output_field/{self.model.it:06}.png')
        close()
        #quit()


    def iterate(self) :
        pass



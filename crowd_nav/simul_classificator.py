import numpy as np
import torch
import torch.nn as nn
import os


SEED = 1000

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# set the random seed for reproducible results
torch.manual_seed(SEED)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # this just calls the base class constructor
        super().__init__()
        # neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically
        
        hidden_size_l1 = 80
        hidden_size_l2 = 40
        hidden_size_l3 = 20

        self.gru = nn.GRU(input_size, hidden_size, num_layers = 3, dropout = 0.5, batch_first=True)
        self.layer1 = nn.Linear(hidden_size, hidden_size_l1)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size_l1, hidden_size_l2)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size_l2, hidden_size_l3)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size_l3, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # the RNN also returns its hidden state but we don't use it
        # while the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default
        x = self.gru(x)[0]
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

class TrackletsClassificator:
    def __init__(self, input_size = 4, hidden_size = 80, output_size = 1, model = None, model_name = 'model_soc_class.pth', device = None):
        """
        The class is to classify the tracklets as social and unsocial. 
        """
        self.device = None
        
        # the NN model
        if model == None:
            self.model = SimpleRNN(input_size, hidden_size, output_size)
        else:
            self.model == model
        self.model = self.model.float()

        if model_name != None:
            self.load_model(model_path = model_name, device = device)
        self.criterion = nn.BCELoss()
        self.optimizer   = torch.optim.RMSprop(self.model.parameters(), lr=0.001)

        print(self.model)

    def load_model(self, model_path, device = None):
        # set the device and load the weights of the NN
        # automatically determine the device that PyTorch should use for computation
        if device == None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # download the model
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        # move model to the device which will be used for the prediction
        self.model.to(self.device)
    
    def test(self, data_array):
        # normalization of the input data
        base_array = np.tile(data_array[0][0], (data_array.shape[1], 1))
        data_array = np.subtract(data_array, base_array)
        # set the model to evaluation mode. This will turn off layers that would
        # otherwise behave differently during training, such as dropout
        self.model.eval()

        # a context manager is used to disable gradient calculations during inference
        # to reduce memory usage, as we typically don't need the gradients at this point
        with torch.no_grad():

            # convert ndarray to Tensor
            data = torch.from_numpy(data_array)
            data = data.float().to(self.device)

            output = self.model(data)
            # pick only the output corresponding to last sequence element (input is pre padded)
            output = output[:, -1, :]

            if output.size()[0] != 1:
                output = torch.squeeze(output)
                output = torch.unsqueeze(output,0)

            y_pred = output.round()

        return y_pred
    
    def train(self, data_list):
        labels = data_list[0]
        data_array = data_list[1]
        # normalization of the input data
        base_array = np.tile(data_array[0][0], (data_array.shape[1], 1))
        data_array = np.subtract(data_array, base_array)
        # set the model to training mode. This will turn on layers that would
        # otherwise behave differently during evaluation, such as dropout
        self.model.train()

        # a context manager is used to disable gradient calculations during inference
        # to reduce memory usage, as we typically don't need the gradients at this point.

        # convert ndarray to Tensor
        data = torch.from_numpy(data_array)
        data = data.float().to(self.device)
        # perform the forward pass of the model
        output = self.model(data)  # Step ①
        # pick only the output corresponding to last sequence element (input is pre padded)
        output = output[:, -1, :]

        # compute the value of the loss for this batch. For loss functions like CrossEntropyLoss,
        # the second argument is actually expected to be a tensor of class indices rather than
        # one-hot encoded class labels. One approach is to take advantage of the one-hot encoding
        # of the target and call argmax along its second dimension to create a tensor of shape
        # (batch_size) containing the index of the class label that was hot for each sequence.
        # target = target.argmax(dim=1)  # For example, [0,1,0,0] will correspond to 1 (index start from 0)  
        if output.size()[0] != 1:
            output = torch.squeeze(output)
            output = torch.unsqueeze(output,0)
        target = torch.tensor(labels)
        target = target.float().to(self.device)
        target = torch.unsqueeze(target,0)
        loss = self.criterion(output, target)  # Step ②

        # clear the gradient buffers of the optimized parameters
        # otherwise, gradients from the previous batch would be accumulated
        self.optimizer.zero_grad()  # Step ③

        loss.backward()  # Step ④

        self.optimizer.step()  # Step ⑤

        y_pred = output.round()

        return y_pred
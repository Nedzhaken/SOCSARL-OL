import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

SEED = 1000

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# set the random seed for reproducible results
torch.manual_seed(SEED)

class TrackletsDataset(Dataset):
    """Tracklets dataset."""
    def __init__(self, folder, transform=None):
        """
        Arguments:
            folder (string): Directory with all .csv tracklet files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # the name of folder with prepared dataset .csv file
        self.folder = folder
        self.transform = transform
        # the list of .csv file names
        self.csv_names = None
        # the list of tracklet dataframes
        self.df_list = []
        # the main dataframe with all tracklets
        self.main_df = self.download_tracklets(self.folder)

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # convert the object list to the np.array
        tracklet = self.main_df.iloc[idx, 1:]        
        tracklet = self.seria_to_list(tracklet)
        input_dim = len(tracklet[0])
        tracklet = np.array([tracklet], dtype=float).reshape(-1, input_dim)
        # get the label
        label = self.main_df.iloc[idx, 0]
        sample = {'tracklet': tracklet, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def seria_to_list(self, seria):
        """
        Transform the seria '[123, 123]' to [123, 123]. 
        """
        str_list = [i for i in seria]
        str_list_cor = []
        for i in str_list:
            i = i.replace('[', '')
            # define the mapping table
            mapping_table = str.maketrans({'[': '', ']': '', ',': ''})
            # use translate() method to replace characters
            i = i.translate(mapping_table)
            value_str = list(i.split(" "))
            point = [float(number) for number in value_str]
            str_list_cor.append(point)
        return str_list_cor

    def download_tracklets(self, folder):
        """
        Load the whole prepared datasets from the folder. 
        """
        # list to store files
        self.csv_names = []
        # iterate directory
        for file_path in os.listdir(folder):
            # check if current file_path is a file
            if os.path.isfile(os.path.join(folder, file_path)):
                if 'tracklets_THOR-Magni' in file_path:
                    # add filename to list
                    self.csv_names.append(file_path)
        for file_name_db in self.csv_names:
            self.load_db(file_name_db)
        # create one main dataframe
        main_df = self.df_list[0]
        for df in self.df_list[1:]:
            main_df = main_df.append(df, ignore_index=True)   
        # convert the Type field to the number
        main_df['Type'] = main_df.Type.astype('category').cat.codes
        return main_df

    def load_db(self, name, nrows=None):
        """
        Create the DataFrame() from .csv file. 
        """
        file_name = self.folder + '/' + name
        df = pd.read_csv(file_name, nrows=nrows)        
        df.drop(df.columns[0], axis = 1, inplace=True)          
        self.df_list.append(df)

class TrackletNormalization(object):
    """Normalize the tracklets to (0, 0) coordinates."""

    def __call__(self, sample):
        # the coordinates of the first point of tracket
        base_array = np.tile(sample['tracklet'][0], (sample['tracklet'].shape[0], 1))
        # substract the first point from all points of tracklet
        new_tracklet = np.subtract(sample['tracklet'], base_array)
        # save a modified tracklet
        new_sample = {'tracklet': new_tracklet, 'label': sample['label']}
        return new_sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tracklet, label = sample['tracklet'], sample['label']

        return {'tracklet': torch.from_numpy(tracklet),
                'label': torch.tensor(label, dtype=torch.int8)}

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
    def __init__(self, folder = 'tracklets', input_size = 4, hidden_size = 80, output_size = 1, model = None, norm = False):
        """
        The class is to classify the tracklets as social and non-social. 
        """
        # the name of folder with prepared tracklet dataset as .csv file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(script_dir, folder)
        # the torch tracklets dataset
        if norm:
            self.dataset = TrackletsDataset(folder, transforms.Compose([TrackletNormalization(), ToTensor()]))
        else:
            self.dataset = TrackletsDataset(folder, transforms.Compose([ToTensor()]))
        self.main_df = self.dataset.main_df
        # the NN model
        if model == None:
            self.model = SimpleRNN(input_size, hidden_size, output_size)
        else:
            self.model == model
        self.model = self.model.float()
        print(self.model)

    def train(self, train_data_gen, criterion, optimizer, device):
        """
        Train function of the social module. 
        """
        # set the model to training mode. This will turn on layers that would
        # otherwise behave differently during evaluation, such as dropout
        self.model.train()

        # store the number of sequences that were classified correctly
        num_correct = 0

        # iterate over every batch of sequences. Note that the length of a data generator
        # is defined as the number of batches required to produce a total of roughly 1000
        # sequences given a batch size
        # for batch_idx in range(len(train_data_gen)):
        for sample_batched in train_data_gen:

            # request a batch of sequences and class labels, convert them into tensors
            # of the correct type, and then send them to the appropriate device
            data, target = sample_batched['tracklet'], sample_batched['label']
            data, target = data.float().to(device), target.float().to(device)
            # perform the forward pass of the model
            output = self.model(data)  # Step ①

            # pick only the output corresponding to last sequence element (input is pre padded)
            output = output[:, -1, :] # for many-to-one RNN architecture, we need output from last RNN cell only

            # compute the value of the loss for this batch. For loss functions like CrossEntropyLoss,
            # the second argument is actually expected to be a tensor of class indices rather than
            # one-hot encoded class labels. One approach is to take advantage of the one-hot encoding
            # of the target and call argmax along its second dimension to create a tensor of shape
            # (batch_size) containing the index of the class label that was hot for each sequence
            if output.size()[0] != 1:
                output = torch.squeeze(output)
                output = torch.unsqueeze(output,0)
            target = torch.unsqueeze(target,0)
            loss = criterion(output, target)  # Step ②

            # clear the gradient buffers of the optimized parameters.
            # otherwise, gradients from the previous batch would be accumulated
            optimizer.zero_grad()  # Step ③

            loss.backward()  # Step ④

            optimizer.step()  # Step ⑤

            y_pred = output.round()
            num_correct += (y_pred == target).sum().item()

        return num_correct, loss.item()

    def test(self, test_data_gen, criterion, device):
        """
        Test function of the social module. 
        """
        # set the model to evaluation mode. This will turn off layers that would
        # otherwise behave differently during training, such as dropout
        self.model.eval()

        # store the number of sequences that were classified correctly
        num_correct = 0

        # a context manager is used to disable gradient calculations during inference
        # to reduce memory usage, as we typically don't need the gradients at this point
        with torch.no_grad():
            for sample_batched in test_data_gen:
                
                data, target = sample_batched['tracklet'], sample_batched['label']
                data, target = data.float().to(device), target.float().to(device)

                output = self.model(data)
                # pick only the output corresponding to last sequence element (input is pre padded)
                output = output[:, -1, :]

                if output.size()[0] != 1:
                    output = torch.squeeze(output)
                    output = torch.unsqueeze(output,0)
                target = torch.unsqueeze(target,0)
                loss = criterion(output, target)

                y_pred = output.round()
                num_correct += (y_pred == target).sum().item()

        return num_correct, loss.item()

    def train_and_test(self, train_data_gen, test_data_gen, criterion, optimizer, max_epochs, verbose=True):
        """
        Trains, tests, and saves the social module. Additionally, it generates visualizations of loss and accuracy results.
        """
        # automatically determine the device that PyTorch should use for computation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        # move model to the device which will be used for train and test
        self.model.to(device)

        # track the value of the loss function and model accuracy across epochs
        history_train = {'loss': [], 'acc': []}
        history_test = {'loss': [], 'acc': []}

        for epoch in range(max_epochs):

            # run the training loop and calculate the accuracy
            # remember that the length of a data generator is the number of batches,
            # so we multiply it by the batch size to recover the total number of sequences
            num_correct, loss = self.train(train_data_gen, criterion, optimizer, device)
            accuracy = float(num_correct) / (len(train_data_gen) * train_data_gen.batch_size) * 100
            history_train['loss'].append(loss)
            history_train['acc'].append(accuracy)

            # do the same for the testing loop
            num_correct, loss = self.test(test_data_gen, criterion, device)
            accuracy = float(num_correct) / (len(test_data_gen) * test_data_gen.batch_size) * 100
            history_test['loss'].append(loss)
            history_test['acc'].append(accuracy)

            if verbose or epoch + 1 == max_epochs:
                print(f'[Epoch {epoch + 1}/{max_epochs}]'
                    f" loss: {history_train['loss'][-1]:.4f}, acc: {history_train['acc'][-1]:2.2f}%"
                    f" - test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")
                
        model_path = f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        torch.save(self.model.state_dict(), model_path)

        # generate diagnostic plots for the loss and accuracy
        _ , axes = plt.subplots(ncols=2, figsize=(9, 4.5))
        for ax, metric in zip(axes, ['loss', 'acc']):
            ax.plot(history_train[metric])
            ax.plot(history_test[metric])
            ax.set_xlabel('epoch', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend(['Train', 'Test'], loc='best')
        plt.show()

        return self.model

    def load_and_test(self, test_data_gen, criterion, model_path = 'model.pth'):
        """
        Load the weights of the social module and calcualte the test loss-accuracy.
        """
        # automatically determine the device that PyTorch should use for computation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        # download the model
        self.model.load_state_dict(torch.load(model_path))

        # move model to the device which will be used for train and test
        self.model.to(device)

        # track the value of the loss function and model accuracy across epochs
        history_test = {'loss': [], 'acc': []}


        # run the testing loop and calculate the accuracy
        # remember that the length of a data generator is the number of batches,
        # so we multiply it by the batch size to recover the total number of sequences
        num_correct, loss = self.test(test_data_gen, criterion, device)
        accuracy = float(num_correct) / (len(test_data_gen) * test_data_gen.batch_size) * 100
        history_test['loss'].append(loss)
        history_test['acc'].append(accuracy)

        print(f"test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")

        return self.model

    def train_and_test_k_fold(self, dataset, criterion, max_epochs, k_folds = 5, verbose=True):
        """
        Trains and tests the social module using k-fold cross-validation.
        """
        # initialize the k-fold cross validation
        kf = KFold(n_splits=k_folds, shuffle=True)

        # automatically determine the device that PyTorch should use for computation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
        print(device)

        k_fold_test = []
        start_time = time.time()
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)): 

            self.model = SimpleRNN(4, 16, 1)
            # move model to the device which will be used for train and test
            self.model.to(device)
            optimizer   = torch.optim.RMSprop(self.model.parameters(), lr=0.001)

            print(f"Fold {fold + 1}")
            print("-------")
            # define the data loaders for the current fold
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            )
            test_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(test_idx),
            )

            # track the value of the loss function and model accuracy across epochs
            history_train = {'loss': [], 'acc': []}
            history_test = {'loss': [], 'acc': []}

            for epoch in range(max_epochs):

                # run the training loop and calculate the accuracy
                # remember that the length of a data generator is the number of batches,
                # so we multiply it by the batch size to recover the total number of sequences
                num_correct, loss = self.train(train_loader, criterion, optimizer, device)
                accuracy = float(num_correct) / (len(train_loader) * train_loader.batch_size) * 100
                history_train['loss'].append(loss)
                history_train['acc'].append(accuracy)

                # do the same for the testing loop
                num_correct, loss = self.test(test_loader, criterion, device)
                accuracy = float(num_correct) / (len(test_loader) * test_loader.batch_size) * 100
                history_test['loss'].append(loss)
                history_test['acc'].append(accuracy)

                if verbose or epoch + 1 == max_epochs:
                    print(f'{time.time() - start_time:.4f}: [Epoch {epoch + 1}/{max_epochs}]'
                        f" loss: {history_train['loss'][-1]:.4f}, acc: {history_train['acc'][-1]:2.2f}%"
                        f" - test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")

            k_fold_test.append(history_test['acc'][-1])
            print(f'{time.time() - start_time:.4f}: [Fold {fold + 1}/{k_folds}]'
                        f" - test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")

        print(f"{time.time() - start_time:.4f}: Final test acc {np.mean(k_fold_test):2.2f}%")
        return self.model

classificator = TrackletsClassificator(folder='tracklets_4s_4hz_v', hidden_size = 16, norm=True)

# setup the training and test data generators
batch_size = 32

train_size = int(0.7 * len(classificator.dataset))
test_size = len(classificator.dataset) - train_size

train_data, test_data = torch.utils.data.random_split(classificator.dataset, [train_size, test_size],
                                                                    torch.Generator().manual_seed(SEED))

train_dataloader = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                        shuffle=True, num_workers=0, pin_memory=True)

# setup the RNN and training settings
criterion   = nn.BCELoss()
optimizer   = torch.optim.RMSprop(classificator.model.parameters(), lr=0.001)
max_epochs  = 1

print('batch_size ' + str(batch_size))
print(criterion)
print(optimizer)

# train the model
model = classificator.train_and_test(train_dataloader, test_dataloader, criterion, optimizer, max_epochs)
# model = classificator.train_and_test_k_fold(classificator.dataset, criterion, max_epochs)
# model = classificator.load_and_test(test_dataloader, criterion, model_path = 'model_2025-03-10_20-22-38.pth')


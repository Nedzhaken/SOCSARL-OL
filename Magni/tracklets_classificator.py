import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from tracklets_dataset import TrackletsDataset, TrackletNormalization, ToTensor
from social_rnn import SocialRNN

SEED = 1000

# set the random seed for reproducible results
torch.manual_seed(SEED)

class TrackletsClassificator:
    """
    Classifier for labeling tracklets as social or non-social.
    """
    def __init__(self, data_folder_name: str = 'tracklets', input_size: int = 4, hidden_size: int = 80, output_size: int = 1,
                 model: nn.Module = None, normalization: bool = False):
        """
        Initialize the tracklet classifier.

        Args:
            data_folder_name (str): Name of the folder containing tracklet data.
            input_size (int): Input feature size for the model.
            hidden_size (int): Hidden layer size of the model.
            output_size (int): Output size of the model.
            model (nn.Module, optional): Pre-initialized model.
            normalization (bool): Whether to apply tracklet normalization.
        """
        script_directory = os.path.dirname(os.path.abspath(__file__))
        data_folder_path = os.path.join(script_directory, data_folder_name)

        if normalization:
            self.dataset = TrackletsDataset(data_folder_path, transforms.Compose([TrackletNormalization(), ToTensor()]))
        else:
            self.dataset = TrackletsDataset(data_folder_path, transforms.Compose([ToTensor()]))

        self.main_df = self.dataset.get_dataframe_all_trajectories()

        if model is None:
            self.model = SocialRNN(input_size, hidden_size, output_size)
        else:
            self.model = model
        self.model = self.model.float()
        print(self.model)

    def train_and_test(self, train_data_gen, test_data_gen, criterion, optimizer, max_epochs: int, verbose: bool = True):
        """
        Trains, tests, and saves the social module. Additionally, it generates visualizations of loss and accuracy results.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print(device)

        history_train = {'loss': [], 'acc': []}
        history_test = {'loss': [], 'acc': []}

        for epoch in range(max_epochs):

            # run the training loop and calculate the accuracy
            # remember that the length of a data generator is the number of batches,
            # so we multiply it by the batch size to recover the total number of sequences
            num_correct, loss = self.train(train_data_gen, criterion, optimizer, device)
            self.acc_loss_calculate(num_correct, loss, train_data_gen, history_train)

            # do the same for the testing loop
            num_correct, loss = self.test(test_data_gen, criterion, device)
            self.acc_loss_calculate(num_correct, loss, test_data_gen, history_test)

            if verbose or epoch + 1 == max_epochs:
                print(f'[Epoch {epoch + 1}/{max_epochs}]'
                    f" loss: {history_train['loss'][-1]:.4f}, acc: {history_train['acc'][-1]:2.2f}%"
                    f" - test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")

        self.save_model()

        self.plot_learning_result(history_train, history_test)

        # generate diagnostic plots for the loss and accuracy

        return self.model

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

    def acc_loss_calculate(self, num_correct: int, loss: float, data_gen: DataLoader, history: dict):
        """
        Calculate accuracy and save accuracy and loss in history dictionary. 
        """
        accuracy = float(num_correct) / (len(data_gen) * data_gen.batch_size) * 100
        history['loss'].append(loss)
        history['acc'].append(accuracy)

    def save_model(self):
        """
        Save the current version of model. 
        """
        model_path = f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        torch.save(self.model.state_dict(), model_path)

    def plot_learning_result(self, history_train: dict, history_test: dict):
        """
        Generate diagnostic plots for the loss and accuracy. 
        """
        _ , axes = plt.subplots(ncols=2, figsize=(9, 4.5))
        for ax, metric in zip(axes, ['loss', 'acc']):
            ax.plot(history_train[metric])
            ax.plot(history_test[metric])
            ax.set_xlabel('epoch', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend(['Train', 'Test'], loc='best')
        plt.show()

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
        self.acc_loss_calculate(num_correct, loss, test_data_gen, history_test)

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

            self.model = SocialRNN(4, 16, 1)
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
                self.acc_loss_calculate(num_correct, loss, train_loader, history_train)

                # do the same for the testing loop
                num_correct, loss = self.test(test_loader, criterion, device)
                self.acc_loss_calculate(num_correct, loss, test_loader, history_test)

                if verbose or epoch + 1 == max_epochs:
                    print(f'{time.time() - start_time:.4f}: [Epoch {epoch + 1}/{max_epochs}]'
                        f" loss: {history_train['loss'][-1]:.4f}, acc: {history_train['acc'][-1]:2.2f}%"
                        f" - test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")

            k_fold_test.append(history_test['acc'][-1])
            print(f'{time.time() - start_time:.4f}: [Fold {fold + 1}/{k_folds}]'
                        f" - test_loss: {history_test['loss'][-1]:.4f}, test_acc: {history_test['acc'][-1]:2.2f}%")

        print(f"{time.time() - start_time:.4f}: Final test acc {np.mean(k_fold_test):2.2f}%")
        return self.model
    
if __name__ == "__main__":
    classificator = TrackletsClassificator(data_folder_name = 'tracklets_4s_4hz_v', hidden_size = 16, normalization = True)

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
    max_epochs  = 50

    print('batch_size ' + str(batch_size))
    print(criterion)
    print(optimizer)

    # train the model
    model = classificator.train_and_test(train_dataloader, test_dataloader, criterion, optimizer, max_epochs)
    # model = classificator.train_and_test_k_fold(classificator.dataset, criterion, max_epochs)
    # model = classificator.load_and_test(test_dataloader, criterion, model_path = 'model_2025-03-10_20-22-38.pth')


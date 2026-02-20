import torch.nn as nn

class SocialRNN(nn.Module):
    HIDDEN_SIZE_L1 = 80
    HIDDEN_SIZE_L2 = 40
    HIDDEN_SIZE_L3 = 20

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers = 3, dropout = 0.5, batch_first=True)
        self.layer1 = nn.Linear(hidden_size, self.HIDDEN_SIZE_L1)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(self.HIDDEN_SIZE_L1, self.HIDDEN_SIZE_L2)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(self.HIDDEN_SIZE_L2, self.HIDDEN_SIZE_L3)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(self.HIDDEN_SIZE_L3, output_size)
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


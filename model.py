# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch

class TaxiDriverClassifier(nn.Module):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(TaxiDriverClassifier, self).__init__()

        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        ###########################
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # print("x",x.shape)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # print("h0",h0.shape)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # print("c0",c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(out)
        out = out[:, -1, :]

        out = self.fc(out)
        return out
        ###########################
        
    

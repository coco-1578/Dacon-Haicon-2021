import torch
import torch.nn as nn


class BaseLine(nn.Module):

    def __init__(self, n_features, hidden_size, num_layers=3, bidirectional=True, dropout=0):
        super(BaseLine, self).__init__()
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size if bidirectional else hidden_size, n_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch, seq, params) --> (seq, batch, params)
        x = x.transpose(0, 1)
        self.gru.flatten_parameters()
        outs, _ = self.gru(x)

        # pick the last output (x_t)
        out = self.fc(outs[-1])

        # skip connection with the x0
        out = x[0] + out
        return out

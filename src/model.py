import torch
import torch.nn as nn


class BaseLine2020(nn.Module):

    def __init__(self, n_features, hidden_size, num_layers=3, bidirectional=True, dropout=0, reversed=False):
        super(BaseLine2020, self).__init__()
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size if bidirectional else hidden_size, n_features)
        self.reversed = reversed

    def forward(self, x):
        # (batch, seq, params) --> (seq, batch, params)
        x = x.transpose(0, 1)

        if self.reversed:
            x = torch.from_numpy(x.cpu().numpy()[::-1, :, :].copy()).cuda()

        self.gru.flatten_parameters()
        outs, _ = self.gru(x)

        # pick the last output (x_t)
        out = self.fc(outs[-1])

        # skip connection with the x0
        if self.reversed:
            out = x[-1] + out
        else:
            out = x[0] + out
        return out

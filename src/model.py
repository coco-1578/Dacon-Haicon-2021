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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.2):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, ntoken)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.transpose(0, 1)  # (batch, seq, features) -> (seq, batch, features)
        output = self.transformer_encoder(x, None)
        output = self.decoder(self.relu(output))
        return x[0] + output[-1]  # skip connection


class LSTMEncoder(nn.Module):

    def __init__(self, n_features, hidden_size, num_layers=2, bidirectional=True, dropout=0):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

    def forward(self, x):

        x = x.transpose(0, 1)  # (batch, seq, features) -> (seq, batch, features)
        x_n, (h_n, c_n) = self.lstm(x)
        # return last output
        return x[-1]  # (batch, features)


class LSTMDecoder(nn.Module):

    def __init__(self, window_size, n_features, hidden_size, num_layers=2, bidirectional=True, dropout=0):
        super(LSTMDecoder, self).__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size if bidirectional else hidden_size, n_features)

    def forward(self, x):
        # x'shape -> (batch, features) -> repeat
        x = x.repeat(self.window_size, 1, 1)
        x_n, (h_n, c_n) = self.lstm(x)
        output = self.fc(x_n[-1])

        return output


class LSTMAE(nn.Module):

    def __init__(self, window_size, n_features, hidden_size, num_layers=2, bidirectional=True, dropout=0):
        super(LSTMAE, self).__init__()

        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers, bidirectional, dropout)  # hidden_size = 128
        self.decoder = LSTMDecoder(window_size, n_features, hidden_size * 2, num_layers, bidirectional, dropout)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


class GRUEncoder(nn.Module):

    def __init__(self, n_features, hidden_size, num_layers=1, bidirectional=True, dropout=0):

        super(GRUEncoder, self).__init__()
        self.gru_1 = nn.GRU(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.gru_2 = nn.GRU(input_size=2 * hidden_size,
                            hidden_size=4 * hidden_size,
                            num_layers=num_layers,
                            bidirectional=False,
                            dropout=dropout)

    def forward(self, x):

        x = x.transpose(0, 1)  # (batch, seq, features) -> (seq, batch, features)
        x, _ = self.gru_1(x)
        x, h = self.gru_2(x)
        return x[-1]


class GRUDecoder(nn.Module):

    def __init__(self, window_size, n_features, hidden_size, num_layers=1, bidirectional=True, dropout=0):

        super(GRUDecoder, self).__init__()
        self.window_size = window_size
        self.gru_1 = nn.GRU(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=False,
                            dropout=dropout)
        self.gru_2 = nn.GRU(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=n_features, bias=True)
        self.relu_1 = nn.ReLU()

    def forward(self, x):

        x = x.repeat(self.window_size, 1, 1)
        x, _ = self.gru_1(x)
        x, _ = self.gru_2(x)
        return self.relu_1(self.fc_1(x))


class GRUAE(nn.Module):

    def __init__(self, window_size, n_features, hidden_size, num_layers=1, bidirectional=True, dropout=0):

        super(GRUAE, self).__init__()
        self.encoder = GRUEncoder(n_features, hidden_size, n_features, bidirectional, dropout)
        self.decoder = GRUDecoder(window_size, n_features, hidden_size, n_features, bidirectional, dropout)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':

    window_size = 60
    n_features = 86
    hidden_size = 128

    model = LSTMAE(window_size, n_features, hidden_size)
    print(model)
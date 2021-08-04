import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, window_size, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim
        self.lstm_1 = nn.LSTM(input_size=self.n_features,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=self.hidden_dim,
                              hidden_size=self.embedding_dim,
                              num_layers=1,
                              batch_first=True)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        return x[:, -1, :]


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class Decoder(nn.Module):

    def __init__(self, window_size, n_features=1, input_dim=64):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.lstm_1 = nn.LSTM(input_size=self.input_dim,
                              hidden_size=self.input_dim,
                              num_layers=1,
                              batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=self.input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1,
                              batch_first=True)
        self.output = nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output)

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_dim).repeat(1, self.window_size, 1)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        return self.timedist(x)


class RecurrentAutoencoder(nn.Module):

    def __init__(self, window_size, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(window_size, n_features, embedding_dim)
        self.decoder = Decoder(window_size, n_features, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

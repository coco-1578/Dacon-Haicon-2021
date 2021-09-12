# Train the Haicon model in the background

import random
import dateutil
import easydict

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from TaPR_pkg import etapr

##### Global Variable
SEED = 5252
MAX_EPOCHS = 128
WINDOW_GIVEN = 89
WINDOW_SIZE = 90
N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 2048

TRAIN_PATH = sorted([x for x in Path('/home/salmon21/coco/Competition/HAICON2021/train/').glob('*.csv')])
VALID_PATH = sorted([x for x in Path('/home/salmon21/coco/Competition/HAICON2021/validation/').glob('*.csv')])
TEST_PATH = sorted([x for x in Path('/home/salmon21/coco/Competition/HAICON2021/test/').glob('*.csv')])

TIMESTAMP_FIELD = 'timestamp'
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = 'attack'
SEQUENTIAL_COLUMNS = [
    "C01", "C03", "C04", "C05", "C06", "C07", "C08", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C20", "C21",
    "C23", "C24", "C25", "C27", "C28", "C30", "C31", "C32", "C33", "C34", "C35", "C37", "C40", "C41", "C42", "C43",
    "C44", "C45", "C46", "C47", "C48", "C50", "C51", "C53", "C54", "C56", "C57", "C58", "C59", "C60", "C61", "C62",
    "C64", "C65", "C66", "C67", "C68", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", "C78", "C79", "C80",
    "C81", "C83", "C84", "C86"
]
CATEGORICAL_COLUMNS = [
    "C02", "C09", "C10", "C18", "C19", "C22", "C26", "C29", "C36", "C38", "C39", "C49", "C52", "C55", "C63", "C69",
    "C82", "C85"
]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#####

##### Fix Seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


##### CSV load function
def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


##### Load CSV
TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_PATH)
VALID_DF_RAW = dataframe_from_csvs(VALID_PATH)
TEST_DF_RAW = dataframe_from_csvs(TEST_PATH)


##### Normalize Dataframe
def normalize(datasets):
    scaler = MinMaxScaler()
    train, valid, test = datasets
    columns = train.columns.drop([TIMESTAMP_FIELD])
    train[columns] = scaler.fit_transform(train[columns])
    valid[columns] = scaler.transform(valid[columns])
    test[columns] = scaler.transform(test[columns])

    return train[columns], valid[columns], test[columns]


TRAIN_DF, VALID_DF, TEST_DF = normalize([TRAIN_DF_RAW, VALID_DF_RAW, TEST_DF_RAW])


##### Define Dataset Class
class HaiconDataset(Dataset):

    def __init__(self, timestamps, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in range(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(self.ts[L]) == timedelta(seconds=WINDOW_SIZE -
                                                                                                  1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)

        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, index):
        i = self.valid_idxs[index]
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i:i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])

        return item


##### Load Dataset
SEQUENTIAL_TRAIN_DATASET = HaiconDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF[SEQUENTIAL_COLUMNS])
CATEGORICAL_TRAIN_DATASET = HaiconDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF[CATEGORICAL_COLUMNS])

SEQUENTIAL_VALID_DATASET = HaiconDataset(VALID_DF_RAW[TIMESTAMP_FIELD], VALID_DF[SEQUENTIAL_COLUMNS])
CATEGORICAL_VALID_DATASET = HaiconDataset(VALID_DF_RAW[TIMESTAMP_FIELD], VALID_DF[CATEGORICAL_COLUMNS])

# SEQUENTIAL_TEST_DATASET = HaiconDataset(TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF[SEQUENTIAL_COLUMNS])
# CATEGORICAL_TEST_DATASET = HaiconDataset(TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF[CATEGORICAL_COLUMNS])


#### Define Model Class
class CategoricalStackedGRU(nn.Module):

    def __init__(self, n_features):
        super(CategoricalStackedGRU, self).__init__()
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=N_HIDDENS,
                          num_layers=N_LAYERS,
                          bidirectional=True,
                          dropout=0)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(N_HIDDENS * 2, n_features)

    def forward(self, x):
        x = x.transpose(0, 1)
        self.gru.flatten_parameters()
        xs, _ = self.gru(x)
        out = self.sigmoid(self.fc[xs[-1]])
        return x[0] + out


def get_sinusoid_encoding_table(n_seq, hidn):

    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / hidn)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table


def get_attn_decoder_mask(seq):
    batch, window_size, d_hidn = seq.size()
    subsequent_mask = torch.ones((batch, window_size, window_size), device=seq.device)
    subsequent_mask = subsequent_mask.triu(diagonal=1)
    return subsequent_mask


class scaleddotproductattention(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout)
        self.scale = 1 / (self.args.d_head**0.5)

    def forward(self, q, k, v, attn_mask=False):
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores.mul_(self.scale)

        if attn_mask is not False:
            scores.masked_fill_(attn_mask, -1e9)
            attn_prob = nn.Softmax(dim=-1)(scores)
            attn_prob = self.dropout(attn_prob)
            context = torch.matmul(attn_prob, v)
        else:
            attn_prob = nn.Softmax(dim=-1)(scores)
            attn_prob = self.dropout(attn_prob)
            context = torch.matmul(attn_prob, v)

        return context, attn_prob


class multiheadattention(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.W_Q = nn.Linear(self.args.d_hidn, self.args.n_head * self.args.d_head)
        self.W_K = nn.Linear(self.args.d_hidn, self.args.n_head * self.args.d_head)
        self.W_V = nn.Linear(self.args.d_hidn, self.args.n_head * self.args.d_head)
        self.scaled_dot_attn = scaleddotproductattention(self.args)
        self.linear = nn.Linear(self.args.n_head * self.args.d_head, self.args.d_hidn)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, q, k, v, attn_mask=False):
        batch_size = q.size(0)
        q_s = self.W_Q(q).view(batch_size, -1, self.args.n_head, self.args.d_head).transpose(1, 2)
        k_s = self.W_K(k).view(batch_size, -1, self.args.n_head, self.args.d_head).transpose(1, 2)
        v_s = self.W_V(v).view(batch_size, -1, self.args.n_head, self.args.d_head).transpose(1, 2)

        if attn_mask is not False:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_head, 1, 1)
            context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        else:
            context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.args.n_head * self.args.d_head)

        output = self.linear(context)
        output = self.dropout(output)

        return output, attn_prob


class poswisefeedforwardnet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv1d(in_channels=self.args.d_hidn, out_channels=self.args.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.args.d_ff, out_channels=self.args.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, inputs):
        output = self.conv1(inputs.transpose(1, 2).contiguous())
        output = self.active(output)
        output = self.conv2(output).transpose(1, 2).contiguous()
        output = self.dropout(output)

        return output


class encoderlayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.self_attn = multiheadattention(self.args)
        self.pos_ffn = poswisefeedforwardnet(self.args)

    def forward(self, inputs):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
        att_outputs = att_outputs + inputs

        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = ffn_outputs + att_outputs

        return ffn_outputs, attn_prob


class encoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc_emb = nn.Linear(in_features=self.args.window_size, out_features=self.args.d_hidn, bias=False)
        nusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.args.e_features, self.args.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(nusoid_table, freeze=True)
        self.layers = nn.ModuleList([encoderlayer(self.args) for _ in range(self.args.n_layer)])
        self.enc_attn_probs = None

    def forward(self, inputs):
        self.enc_attn_probs = []
        positions = torch.arange(inputs.size(2), device=inputs.device).expand(inputs.size(0),
                                                                              inputs.size(2)).contiguous()
        outputs = self.enc_emb(inputs.transpose(2, 1).contiguous()) + self.pos_emb(positions)

        for layer in self.layers:
            outputs, enc_attn_prob = layer(outputs)
            self.enc_attn_probs.append(enc_attn_prob)

        return outputs


class decoderlayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.self_attn = multiheadattention(self.args)
        self.dec_enc_attn = multiheadattention(self.args)
        self.pos_ffn = poswisefeedforwardnet(self.args)

    def forward(self, dec_inputs, enc_outputs, attn_mask):
        self_att_outputs, dec_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        self_att_outputs = self_att_outputs + dec_inputs

        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs)
        dec_enc_att_outputs = dec_enc_att_outputs + self_att_outputs

        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = ffn_outputs + dec_enc_att_outputs

        return ffn_outputs, dec_attn_prob, dec_enc_attn_prob


class decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dec_emb = nn.Linear(in_features=self.args.d_features, out_features=self.args.d_hidn, bias=False)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.args.window_size, self.args.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([decoderlayer(self.args) for _ in range(self.args.n_layer)])
        self.dec_attn_probs = None
        self.dec_enc_attn_probs = None

    def forward(self, dec_inputs, enc_outputs):
        self.dec_attn_probs = []
        self.dec_enc_attn_probs = []
        positions = torch.arange(dec_inputs.size(1),
                                 device=dec_inputs.device).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous()
        dec_output = self.dec_emb(dec_inputs) + self.pos_emb(positions)

        attn_mask = torch.gt(get_attn_decoder_mask(dec_inputs), 0)

        for layer in self.layers:
            dec_outputs, dec_attn_prob, dec_enc_attn_prob = layer(dec_output, enc_outputs, attn_mask)
            self.dec_attn_probs.append(dec_attn_prob)
            self.dec_enc_attn_probs.append(dec_enc_attn_prob)

        return dec_outputs


class TimeDistributed(nn.Module):

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))

        return y


class Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = encoder(self.args)
        self.decoder = decoder(self.args)
        self.fc1 = TimeDistributed(
            nn.Linear(in_features=self.args.window_size * self.args.d_hidn, out_features=self.args.dense_h))
        self.fc2 = TimeDistributed(nn.Linear(in_features=self.args.dense_h, out_features=self.args.output_size))

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)

        dec_outputs = self.fc1(dec_outputs.view(dec_outputs.size(0), -1))
        dec_outputs = self.fc2(dec_outputs)

        return dec_outputs


##### Define Model, Optimizer, Scheduler, DataLoader, Loss Functions
seq_args = easydict.EasyDict({
    'output_size': TRAIN_DF[SEQUENTIAL_COLUMNS].shape[1],
    'window_size': WINDOW_GIVEN,
    'batch_size': BATCH_SIZE,
    'lr': 1e-3,
    'e_features': TRAIN_DF[SEQUENTIAL_COLUMNS].shape[1],
    'd_features': TRAIN_DF[SEQUENTIAL_COLUMNS].shape[1],
    'd_hidn': 128,
    'n_head': 4,
    'd_head': 32,
    'dropout': 0.1,
    'd_ff': 128,
    'n_layer': 3,
    'dense_h': 128,
})

SEQUANTIAL_MODEL = Transformer(seq_args)
SEQUANTIAL_MODEL.to(DEVICE)

CATEGORICAL_MODEL = CategoricalStackedGRU(n_features=TRAIN_DF[CATEGORICAL_COLUMNS].shape[1])
CATEGORICAL_MODEL.to(DEVICE)

SEQUANTIAL_OPTIMIZER = torch.optim.AdamW(SEQUANTIAL_MODEL.parameters())
SEQUANTIAL_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(SEQUANTIAL_OPTIMIZER, T_max=MAX_EPOCHS)

CATEGORICAL_OPTIMIZER = torch.optim.AdamW(CATEGORICAL_MODEL.parameters())
CATEGORICAL_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(CATEGORICAL_OPTIMIZER, T_max=MAX_EPOCHS)

SEQUANTIAL_LOSS = nn.MSELoss()
CATEGORICAL_LOSS = nn.MSELoss()

SEQUANTIAL_TRAIN_LOADER = DataLoader(SEQUENTIAL_TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
CATEGORICAL_TRAIN_LOADER = DataLoader(CATEGORICAL_TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

##### Train Model
BEST = {'loss': np.inf}
LOSS_HISTORY = []

SEQUANTIAL_MODEL.train()
CATEGORICAL_MODEL.train()

progress_bar = tqdm.trange(MAX_EPOCHS)

for epoch in progress_bar:
    epoch_loss = 0

    for (sequantial_batch, categorical_batch) in zip(SEQUANTIAL_TRAIN_LOADER, CATEGORICAL_TRAIN_LOADER):
        SEQUANTIAL_OPTIMIZER.zero_grad()
        CATEGORICAL_OPTIMIZER.zero_grad()

        sequantial_given = sequantial_batch['given'].to(DEVICE)
        categorical_given = categorical_batch['given'].to(DEVICE)

        sequantial_answer = sequantial_batch['answer'].to(DEVICE)
        categorical_answer = categorical_batch['answer'].to(DEVICE)

        sequantial_guess = SEQUANTIAL_MODEL(sequantial_given, sequantial_given)
        categorical_guess = CATEGORICAL_MODEL(categorical_given)

        sequantial_loss = SEQUANTIAL_LOSS(sequantial_answer, sequantial_given)
        categorical_loss = CATEGORICAL_LOSS(categorical_answer, categorical_given)

        loss = sequantial_loss + categorical_loss
        loss.backward()
        epoch_loss += loss.item()

        SEQUANTIAL_OPTIMIZER.step()
        CATEGORICAL_OPTIMIZER.step()
    progress_bar.set_postfix_str(f"Train Loss: [{epoch_loss:.4f}]")
    LOSS_HISTORY.append(epoch_loss)
    if epoch_loss < BEST['loss']:
        BEST['sequantial_state'] = SEQUANTIAL_MODEL.state_dict()
        BEST['categorical_state'] = CATEGORICAL_MODEL.state_dict()
        BEST['loss'] = epoch_loss
        BEST['epoch'] = epoch + 1

    SEQUANTIAL_SCHEDULER.step()
    CATEGORICAL_SCHEDULER.step()

##### Save and load best model parameters
with open('/home/salmon21/coco/Competition/HAICON2021/result/model.pt', 'wb') as fd:
    torch.save(
        {
            'sequantial_state': BEST['sequantial_state'],
            'categorical_state': BEST['categorical_state'],
            'epoch': BEST['epoch'],
            'loss_history': LOSS_HISTORY
        }, fd)

with open('/home/salmon21/coco/Competition/HAICON2021/result/model.pt', 'rb') as fd:
    BEST_WEIGHT = torch.load(fd, map_location='cpu')

SEQUANTIAL_MODEL.load_state_dict(BEST_WEIGHT['sequantial_state'])
CATEGORICAL_MODEL.load_state_dict(BEST_WEIGHT['categorical_state'])

SEQUANTIAL_MODEL.to(DEVICE)
CATEGORICAL_MODEL.to(DEVICE)

##### Validate Model
SEQUANTIAL_MODEL.eval()
CATEGORICAL_MODEL.eval()

SEQUANTIAL_VALID_LOADER = DataLoader(SEQUENTIAL_VALID_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
CATEGORICAL_VALID_LOADER = DataLoader(CATEGORICAL_VALID_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

sequantial_ts, sequantial_dist, sequantial_att = [], [], []
categorical_ts, categorical_dist, categorical_att = [], [], []

with torch.no_grad():
    for (sequantial_batch, categorical_batch) in zip(SEQUANTIAL_VALID_LOADER, CATEGORICAL_VALID_LOADER):

        sequantial_given = sequantial_batch['given'].to(DEVICE)
        categorical_given = categorical_batch['given'].to(DEVICE)

        sequantial_answer = sequantial_batch['answer'].to(DEVICE)
        categorical_answer = categorical_batch['answer'].to(DEVICE)

        sequantial_guess = SEQUANTIAL_MODEL(sequantial_given, sequantial_given)
        categorical_guess = CATEGORICAL_MODEL(categorical_given)

        sequantial_ts.append(np.array(sequantial_batch['ts']))
        categorical_ts.append(np.array(categorical_batch['ts']))

        sequantial_dist.append(torch.abs(sequantial_answer - sequantial_guess).cpu().numpy())
        categorical_dist.append(torch.abs(categorical_answer - categorical_guess).cpu().numpy())

        try:
            sequantial_att.append(np.array(sequantial_batch['attack']))
            categorical_att.append(np.array(categorical_batch['attack']))
        except:
            sequantial_att.append(np.zeros(BATCH_SIZE))
            categorical_att.append(np.zeros(BATCH_SIZE))

sequantial_ts, sequantial_dist, sequantial_att = np.concatenate(sequantial_ts), np.concatenate(
    sequantial_dist), np.concatenate(sequantial_att)
categorical_ts, categorical_dist, categorical_att = np.concatenate(categorical_ts), np.concatenate(
    categorical_dist), np.concatenate(categorical_att)

SEQUANTIAL_ANONMALY_SCORE = np.mean(sequantial_dist, axis=1)
CATEGORICAL_ANONMALY_SCORE = np.mean(categorical_dist, axis=1)

ANOMALY_SCORE = SEQUANTIAL_ANONMALY_SCORE + CATEGORICAL_ANONMALY_SCORE


##### Define Range Check function (from Daicon)
def range_check(series, size):
    data = []

    for i in range(len(series) - size + 1):
        if i == 0:
            check_std = np.std(series[i:i + size])
        std = np.std(series[i:i + size])
        mean = np.mean(series[i:i + size])
        max = np.max(series[i:i + size])
        if check_std * 2 >= std:
            check_std = std
            data.append(mean)
        elif max == series[i]:
            data.append(max * 5)
            check_std = std
        else:
            data.append(series[i] * 3)
    for _ in range(size - 1):
        data.append(mean)

    return np.array(data)


RC_ANOMALY_SCORE = range_check(ANOMALY_SCORE, 60)


##### Define Graph check function (from Daicon baseline)
def check_graph(xs, att, piece=2, THRESHOLD=None):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if THRESHOLD != None:
            axs[i].axhline(y=THRESHOLD, color='r')
    plt.savefig('/home/salmon21/coco/Competition/HAICON2021/result/validation_result.png')


##### Another function in the Daicon baseline
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs


def fill_blank(check_ts, labels, total_ts):

    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)


##### Get the best threshold in the validation step without range check function
BEST_F1 = 0
BEST_THRESHOLD = 0
VALID_ATTACK_LABEL = put_labels(np.array(VALID_DF_RAW[ATTACK_FIELD]), threshold=0.5)
for threshold in [
        0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03,
        0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04
]:
    labels = put_labels(ANOMALY_SCORE, threshold)
    final_labels = fill_blank(sequantial_ts, labels, np.array(VALID_DF_RAW[TIMESTAMP_FIELD]))
    ta_pr = etapr.evaluate_haicon(anomalies=VALID_ATTACK_LABEL, predictions=final_labels)
    print(f"F1: {ta_pr['f1']:.3f} (TaP: {ta_pr['TaP']:.3f}, TaR: {ta_pr['TaR']:.3f})")
    if ta_pr['f1'] > BEST_F1:
        BEST_F1 = ta_pr['f1']
        BEST_THRESHOLD = threshold

##### Check the RC_ANOMALY_SCORE with the best threshold in validation step
labels = put_labels(RC_ANOMALY_SCORE, BEST_THRESHOLD)
final_labels = fill_blank(sequantial_ts, labels, np.array(VALID_DF_RAW[TIMESTAMP_FIELD]))
ta_pr = etapr.evaluate_haicon(anomalies=VALID_ATTACK_LABEL, predictions=final_labels)
print('RC_ANOMALY_SCORE Result')
print(f"F1: {ta_pr['f1']:.3f} (TaP: {ta_pr['TaP']:.3f}, TaR: {ta_pr['TaR']:.3f})")
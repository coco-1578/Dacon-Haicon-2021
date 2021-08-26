import random
import dateutil

import torch
import numpy as np
import matplotlib.pyplot as plt


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


def fix_seed(seed=5252):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def save_weight(path, state_dict):

    with open(path, "wb") as fd:
        torch.save(state_dict, fd)


def load_weight(path):

    with open(path, "rb") as fd:
        state_dict = torch.load(fd, map_location="cpu")

    return state_dict


def check_graph(xs, att, piece=2, threshold=None, path=None):

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
        if threshold is not None:
            axs[i].axhline(y=threshold, color='r')

    plt.savefig(path)


class EarlyStopping:

    def __init__(self, patience=5, delta=0):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# 2020 HACION
def Cross_put_labels(Inverse1, Inverse2, Inverse3, Inverse4, Inverse5, Inverse6, Inverse7, Inverse8, Inverse9):
    xs = np.zeros_like(Inverse1)
    for i in range(Inverse1.shape[0]):
        if Inverse1[i] + Inverse2[i] + Inverse3[i] + Inverse4[i] + Inverse5[i] + Inverse6[i] + Inverse7[i] + Inverse8[
                i] + Inverse9[i] > 0:
            xs[i] = 1
        else:
            xs[i] = 0
    return xs
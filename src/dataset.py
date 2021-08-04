import dateutil

from datetime import timedelta

import tqdm
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


def normalize_dataset(train_dataset, valid_dataset, test_dataset):

    columns = train_dataset.columns.drop(["time"])
    scaler = MinMaxScaler()
    train_dataset[columns] = scaler.fit_transform(train_dataset[columns])
    valid_dataset[columns] = scaler.transform(valid_dataset[columns])
    test_dataset[columns] = scaler.transform(test_dataset[columns])

    return train_dataset, valid_dataset, test_dataset, columns


def boundary_check(dataset):
    x = np.array(dataset, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))


class LSTMDataset(Dataset):

    def __init__(self, timestamps, dataset, stride=1, window_size=90, attacks=None):

        self.timestamps = np.array(timestamps)
        self.dataset = np.array(dataset, dtype=np.float32)
        self.valid_idxs = list()
        self.window_size = window_size

        for L in tqdm.tqdm(range(len(self.timestamps) - window_size + 1)):
            R = L + window_size - 1
            if dateutil.parser.parse(self.timestamps[R]) - dateutil.parser.parse(
                    self.timestamps[L]) == timedelta(seconds=window_size - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.window_size - 1
        item = {"attack": self.attacks[last] if self.with_attack else {}}
        item["timestamp"] = self.timestamps[last]
        item["inputs"] = torch.from_numpy(self.dataset[i:i + self.window_size - 1])
        item["labels"] = torch.from_numpy(self.dataset[last])
        return item
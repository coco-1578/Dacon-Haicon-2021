import os
import glob

from datetime import timedelta
from dateutil.parser import parse
from sklearn import preprocessing
from torch.utils import data

import tqdm
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, dataset
from sklearn.preprocessing import MinMaxScaler


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


def normalize_datasets(datasets):

    train_datasets, valid_datasets, test_datasets = datasets

    columns = train_datasets.columns.drop(["timestamp"])
    scaler = MinMaxScaler()
    train_datasets[columns] = scaler.fit_transform(train_datasets[columns])
    valid_datasets[columns] = scaler.transform(valid_datasets[columns])
    test_datasets[columns] = scaler.transform(test_datasets[columns])

    return train_datasets, valid_datasets, test_datasets, columns, scaler


def boundary_check(dataset):
    x = np.array(dataset, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))


def load_datasets(directory):

    train_datasets_path = sorted(glob.glob(os.path.join(directory, 'train/*.csv')))
    valid_datasets_path = sorted(glob.glob(os.path.join(directory, 'validation/*.csv')))
    test_datasets_path = sorted(glob.glob(os.path.join(directory, 'test/*.csv')))

    train_datasets = dataframe_from_csvs(train_datasets_path)
    valid_datasets = dataframe_from_csvs(valid_datasets_path)
    test_datasets = dataframe_from_csvs(test_datasets_path)

    train_datasets, valid_datasets, test_datasets, columns, scaler = normalize_datasets(
        [train_datasets, valid_datasets, test_datasets])

    return train_datasets, valid_datasets, test_datasets, columns, scaler


class BaseLineDataset(Dataset):

    def __init__(self, timestamps, data_frame, window_size, stride=1, attacks=None, is_autoencoder=False):

        self.timestamps = np.array(timestamps)
        self.data_frame = np.array(data_frame, dtype=np.float32)
        self.valid_indices = list()
        self.window_size = window_size
        self.is_autoencoder = is_autoencoder

        for left_window in tqdm.tqdm(range(len(self.timestamps) - self.window_size + 1)):
            right_window = left_window + self.window_size - 1
            if parse(self.timestamps[right_window]) - parse(self.timestamps[left_window]) == timedelta(
                    seconds=self.window_size - 1):
                self.valid_indices.append(left_window)

        self.valid_indices = np.array(self.valid_indices, dtype=np.int32)[::stride]
        self.num_indices = len(self.valid_indices)
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.is_attacked = True
        else:
            self.is_attacked = False

    def __len__(self):
        return self.num_indices

    def __getitem__(self, index):
        index = self.valid_indices[index]
        last_index = index + self.window_size - 1
        item = {"attack": self.attacks[last_index]} if self.is_attacked else {}
        item["timestamp"] = self.timestamps[index + self.window_size - 1]
        if not self.is_autoencoder:
            item["inputs"] = torch.from_numpy(self.data_frame[index:index + self.window_size - 1])
            item["labels"] = torch.from_numpy(self.data_frame[last_index])
        else:
            item["inputs"] = torch.from_numpy(self.data_frame[index:index + self.window_size])
            item["labels"] = torch.from_numpy(self.data_frame[index:index + self.window_size])

        return item
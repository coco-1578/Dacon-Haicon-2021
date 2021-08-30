import argparse

import torch.nn as nn
import torch.optim as optim

from config import CFG
from utils import fix_seed
from trainer import Trainer
from model import *
from dataset import load_datasets, BaseLineDataset


def parse_args():

    parser = argparse.ArgumentParser("Haicon")
    parser.add_argument('-d', '--directory', type=str, help='dataset directory path')
    parser.add_argument('-e', '--ensemble', action='store_true')
    args = parser.parse_args()
    return args


def main():

    fix_seed(CFG.SEED)

    args = parse_args()
    train_dataframe, valid_dataframe, test_dataframe, columns, scaler = load_datasets(args.directory)

    for window_size in CFG.WINDOW_SIZE_LIST:

        # Define Dataset
        train_datasets = BaseLineDataset(train_dataframe['timestamp'],
                                         train_dataframe[columns],
                                         window_size,
                                         stride=1,
                                         attacks=None)
        valid_datasets = BaseLineDataset(valid_dataframe['timestamp'],
                                         valid_dataframe[columns],
                                         window_size,
                                         stride=1,
                                         attacks=valid_dataframe['attack'])
        test_datasets = BaseLineDataset(test_dataframe['timestamp'],
                                        test_dataframe[columns],
                                        window_size,
                                        stride=1,
                                        attacks=None)

        model = BaseLine(train_dataframe[columns].shape[1], CFG.HIDDEN_SIZE, CFG.NUM_LAYERS, CFG.BIDIRECTIONAL,
                         CFG.DROPOUT)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.MAX_EPOCHS)
        trainer = Trainer(CFG, model, criterion, optimizer, scheduler, window_size)
        trainer.fit(train_datasets, valid_datasets, valid_dataframe)
        trainer.predict(test_datasets)


if __name__ == '__main__':

    main()
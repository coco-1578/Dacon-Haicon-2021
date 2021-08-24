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
    args = parser.parse_args()
    return args


def main():

    fix_seed(CFG.SEED)

    args = parse_args()

    train_dataframe, valid_dataframe, test_dataframe, columns, scaler = load_datasets(args.directory)

    # model = BaseLine(n_features=train_dataframe.shape[1],
    #                  hidden_size=CFG.HIDDEN_SIZE,
    #                  num_layers=CFG.NUM_LAYERS,
    #                  bidirectional=CFG.BIDIRECTIONAL,
    #                  dropout=CFG.DROPOUT)

    model = LSTMAE(CFG.WINDOW_SIZE, train_dataframe.shape[1], CFG.HIDDEN_SIZE, CFG.NUM_LAYERS, CFG.BIDIRECTIONAL,
                   CFG.DROPOUT)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR, betas=CFG.BETAS, weight_decay=CFG.DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.MAX_EPOCHS, eta_min=1e-5)
    trainer = Trainer(CFG, model, criterion, optimizer, scheduler)
    train_datasets = BaseLineDataset(train_dataframe['timestamp'],
                                     train_dataframe[columns],
                                     CFG.WINDOW_SIZE,
                                     stride=1,
                                     attacks=None)
    valid_datasets = BaseLineDataset(valid_dataframe['timestamp'],
                                     valid_dataframe[columns],
                                     CFG.WINDOW_SIZE,
                                     stride=1,
                                     attacks=valid_dataframe['attack'])
    test_datasets = BaseLineDataset(test_dataframe['timestamp'],
                                    test_dataframe[columns],
                                    CFG.WINDOW_SIZE,
                                    stride=1,
                                    attacks=None)

    # train & validation
    trainer.fit(train_datasets, valid_datasets)

    # test
    trainer.predict(test_datasets)


if __name__ == '__main__':

    main()
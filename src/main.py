import argparse

from config import CFG
from utils import fix_seed
from trainer import Trainer
from model import BaseLine2020
from dataset import load_datasets


def parse_args():

    parser = argparse.ArgumentParser("Haicon")
    parser.add_argument('-d', '--directory', type=str, help='dataset directory path')
    args = parser.parse_args()
    return args


def main():

    fix_seed(CFG.SEED)

    args = parse_args()

    train_datasets, valid_datasets, test_datasets, columns = load_datasets(args.directory)

    model = BaseLine2020(n_features=train_datasets.shape[1],
                         hidden_size=CFG.HIDDEN_SIZE,
                         num_layers=CFG.NUM_LAYERS,
                         bidirectional=CFG.BIDIRECTIONAL,
                         dropout=CFG.DROPOUT,
                         reversed=CFG.REVERSED)
    trainer = Trainer(CFG, model)
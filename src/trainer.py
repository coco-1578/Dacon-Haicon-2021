import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from utils import *
from TaPR_pkg import etapr


class Trainer:

    def __init__(self, CFG, model):

        self.CFG = CFG
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.CFG.LR,
                                     betas=self.CFG.BETAS,
                                     weight_decay=self.CFG.DECAY)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR()

    def _train_on_batch(self, batch):

        self.model.train()

        inputs = batch['inputs'].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self.model(inputs)
        # get the last one
        outputs = outputs[-1]
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _valid_on_epoch(self, valid_loader):

        timestamp, distance, attacks = [], [], []
        valid_losses = 0
        self.model.eval()
        with torch.no_grad():
            for batch in valid_loader:

                inputs = batch['inputs'].cuda()
                labels = batch['labels'].cuda()

                outputs = self.model(inputs)
                # get the last one
                outputs = outputs[-1]
                loss = self.criterion(labels, outputs)
                valid_losses += loss.item()

                timestamp.append(np.array(batch["timestamp"]))
                distance.append(torch.abs(labels - outputs).cpu().numpy())
                try:
                    attacks.append(np.array(batch["attack"]))
                except:
                    attacks.append(np.zeros(self.CFG.BATCH_SIZE))
        return np.concatenate(timestamp), np.concatenate(distance), np.concatenate(attacks), valid_losses

    def run(self, train_dataset, valid_dataset, columns):

        train_loader = DataLoader(train_dataset[columns])
        valid_loader = DataLoader(valid_dataset[columns])

        loss_history = {"train_loss": [], "valid_loss": []}

        for epoch in tqdm.tqdm(range(self.CFG.MAX_EPOCH)):
            train_losses = 0
            progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
            for index, batch in progress_bar:

                train_loss = self._train_on_batch(batch)
                train_losses += train_loss

            # validation process
            timestamp, distance, attacks, valid_losses = self._valid_on_epoch(valid_loader)
            anomaly_score = np.mean(distance, axis=1)

            labels = put_labels(anomaly_score)
            attack_labels = put_labels(np.array(valid_dataset["attack"]), 0.5)
            final_labels = fill_blank(timestamp, labels, np.array(valid_dataset["time"]))

            assert attack_labels.shape[0] == final_labels.shape[0], "Length of the list should be same"
            tapr = etapr.evaluate_haicon(anomalies=attack_labels, predictions=final_labels)

            loss_history['train_loss'].append(train_losses)
            loss_history['valid_loss'].append(valid_losses)

            print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")
            print(f"# of detected anomalies: {len(tapr['Detected_Anomalies'])}")
            print(f"Detected anomalies: {tapr['Detected_Anomalies']}")
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from utils import *
from TaPR_pkg import etapr


class Trainer:

    def __init__(self, CFG, model, criterion, optimizer, scheduler, window_size):

        self.CFG = CFG
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.window_size = window_size

    def _train_on_batch(self, batch):

        self.model.train()

        inputs = batch['inputs'].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self.model(inputs)
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

                inputs = batch['inputs'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(labels, outputs)
                valid_losses += loss.item()

                timestamp.append(np.array(batch["timestamp"]))
                distance.append(torch.abs(labels - outputs).cpu().numpy())
                try:
                    attacks.append(np.array(batch["attack"]))
                except:
                    attacks.append(np.zeros(self.CFG.BATCH_SIZE))

        return np.concatenate(timestamp), np.concatenate(distance), np.concatenate(attacks), valid_losses

    def _predict(self, test_loader):

        timestamp, distance, attacks = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:

                inputs = batch['inputs'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(inputs)

                timestamp.append(np.array(batch['timestamp']))
                distance.append(torch.abs(labels - outputs).cpu().numpy())
                try:
                    attacks.append(np.array(batch['attack']))
                except:
                    attacks.append(np.zeros(self.CFG.BATCH_SIZE))

        return np.concatenate(timestamp), np.concatenate(distance), np.concatenate(attacks)

    def fit(self, train_dataset, valid_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.CFG.BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=self.CFG.BATCH_SIZE, shuffle=False, num_workers=4)

        early_stopping = EarlyStopping(patience=10)

        loss_history = {"train_loss": [], "valid_loss": []}
        best = {"valid_loss": np.inf}

        for epoch in tqdm.tqdm(range(self.CFG.MAX_EPOCHS)):
            train_losses = 0
            progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
            for index, batch in progress_bar:

                train_loss = self._train_on_batch(batch)
                train_losses += train_loss

                description = f"Epoch: [{epoch+1}] - Train Loss: [{train_losses:.4f}]"
                progress_bar.set_description(description)

            # validation step per epoch
            timestamp, distance, attacks, valid_losses = self._valid_on_epoch(valid_loader)
            anomaly_score = np.mean(distance, axis=1)

            labels = put_labels(anomaly_score)
            attack_labels = put_labels(np.array(valid_dataset["attack"]), 0.5)
            final_labels = fill_blank(timestamp, labels, np.array(valid_dataset["time"]))

            assert attack_labels.shape[0] == final_labels.shape[0], "Length of the list should be same"
            tapr = etapr.evaluate_haicon(anomalies=attack_labels, predictions=final_labels)

            loss_history["train_loss"].append(train_losses)
            loss_history['valid_loss'].append(valid_losses)
            print(f"Valid Loss: [{valid_losses:.4f}]")
            print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")

            if valid_losses < best["valid_loss"]:
                best["valid_loss"] = valid_losses
                best["state"] = self.model.state_dict()

            early_stopping(valid_losses)
            if early_stopping.early_stop:
                break

        # TODO: save best train model and load best weight
        save_weight(f"result/{self.window_size}_{self.CFG.NUM_LAYERS}_{self.CFG.HIDDEN_SIZE}.pth", best["state"])
        state_dict = load_weight(f"result/{self.window_size}_{self.CFG.NUM_LAYERS}_{self.CFG.HIDDEN_SIZE}.pth")
        self.model.load_state_dict(state_dict).to(self.device)

        # Final Validation process
        timestamp, distance, attacks, valid_losses = self._valid_on_epoch(valid_loader)
        anomaly_score = np.mean(distance, axis=1)

        labels = put_labels(anomaly_score)
        attack_labels = put_labels(np.array(valid_dataset["attack"]), 0.5)
        final_labels = fill_blank(timestamp, labels, np.array(valid_dataset["time"]))

        assert attack_labels.shape[0] == final_labels.shape[0], "Length of the list should be same"
        tapr = etapr.evaluate_haicon(anomalies=attack_labels, predictions=final_labels)
        print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")
        print(f"# of detected anomalies: {len(tapr['Detected_Anomalies'])}")
        print(f"Detected anomalies: {tapr['Detected_Anomalies']}")

        check_graph(anomaly_score,
                    attacks,
                    threshold=0.04,
                    path=f"result/{self.window_size}_{self.CFG.NUM_LAYERS}_{self.CFG.HIDDEN_SIZE}.png")

    def predict(self, test_dataset):

        test_loader = DataLoader(test_dataset, batch_size=self.CFG.BATCH_SIZE, shuffle=False, num_workers=4)

        # load best weight
        state_dict = load_weight(f"result/{self.window_size}_{self.CFG.NUM_LAYERS}_{self.CFG.HIDDEN_SIZE}.pth")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        timestamps, distance, attacks = self._predict(test_loader)
        anomaly_score = np.mean(distance, axis=1)

        labels = put_labels(anomaly_score)

        submission = pd.read_csv('sample_submission.csv')
        submission.index = submission['timestamp']
        submission.loc[timestamps, 'attack'] = labels

        submission.to_csv(f"result/{self.window_size}_{self.CFG.NUM_LAYERS}_{self.CFG.HIDDEN_SIZE}_submission.csv",
                          index=False)

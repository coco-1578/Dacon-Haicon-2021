import tqdm
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from utils import *
from config import CFG
from TaPR_pkg import etapr


class Trainer:

    def __init__(self, model, criterion, optimizer, scheduler, window_size):

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

    def _predict(self, data_loader):

        timestamp, distance, attacks = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:

                inputs = batch['inputs'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(inputs)

                timestamp.append(np.array(batch['timestamp']))
                distance.append(torch.abs(labels - outputs).cpu().numpy())
                try:
                    attacks.append(np.array(batch['attack']))
                except:
                    attacks.append(np.zeros(CFG.BATCH_SIZE))

        return np.concatenate(timestamp), np.concatenate(distance), np.concatenate(attacks)

    def fit(self, train_dataset, valid_dataset, valid_dataframe):

        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)

        early_stopping = EarlyStopping(patience=10)

        loss_history = {"train_loss": [], "valid_loss": []}
        best = {"train_loss": np.inf, "taf1": -np.inf}

        epochs = tqdm.trange(CFG.MAX_EPOCHS)
        for epoch in epochs:
            train_losses = 0
            for batch in train_loader:

                train_loss = self._train_on_batch(batch)
                train_losses += train_loss

            epochs.set_postfix_str(f"Train Loss: [{train_losses:.4f}]")

            if train_losses < best["train_loss"]:
                best["epoch"] = epoch
                best["train_loss"] = train_losses
                best["state"] = self.model.state_dict()

            self.scheduler.step()
            early_stopping(train_losses)
            if early_stopping.early_stop:
                break

        save_weight(f"result/{self.window_size}_{CFG.NUM_LAYERS}_{CFG.HIDDEN_SIZE}.pth", best)
        state_dict = load_weight(f"result/{self.window_size}_{CFG.NUM_LAYERS}_{CFG.HIDDEN_SIZE}.pth")
        self.model.load_state_dict(state_dict['state'])
        self.model = self.model.to(self.device)

        # Validation Process
        timestamp, distance, attacks, = self._predict(valid_loader)
        anomaly_score = np.mean(distance, axis=1)

        # Find proper threshold
        for threshold in CFG.THRESHOLD_RANGE:
            labels = put_labels(anomaly_score, CFG.THRESHOLD)
            attack_labels = put_labels(np.array(valid_dataframe["attack"]), 0.5)
            final_labels = fill_blank(timestamp, labels, np.array(valid_dataframe["timestamp"]))

            assert attack_labels.shape[0] == final_labels.shape[0], "Length of the list should be same"
            tapr = etapr.evaluate_haicon(anomalies=attack_labels, predictions=final_labels)
            if tapr['f1'] > best['taf1']:
                best['taf1'] = tapr['f1']
                best['tap'] = tapr['TaP']
                best['tar'] = tapr['TaR']
                best['threshold'] = threshold

        print(
            f"Threshold: {best['threshold']}, F1: {best['taf1']:.3f} (TaP: {best['tap']:.3f}, TaR: {best['tar']:.3f})")
        c_anomaly_score = range_check(anomaly_score, 60)
        labels = put_labels(c_anomaly_score, CFG.THRESHOLD)
        attack_labels = put_labels(np.array(valid_dataframe["attack"]), 0.5)
        final_labels = fill_blank(timestamp, labels, np.array(valid_dataframe["timestamp"]))
        tapr = etapr.evaluate_haicon(anomalies=attack_labels, predictions=final_labels)
        print(f"F1: {tapr['f1']:.3f} (TaP: {tapr['TaP']:.3f}, TaR: {tapr['TaR']:.3f})")

        # print(f"# of detected anomalies: {len(tapr['Detected_Anomalies'])}")
        # print(f"Detected anomalies: {tapr['Detected_Anomalies']}")

        save_weight(f"result/{self.window_size}_{CFG.NUM_LAYERS}_{CFG.HIDDEN_SIZE}.pth", best)
        check_graph(c_anomaly_score,
                    attacks,
                    piece=5,
                    threshold=best['threshold'],
                    path=f"result/{self.window_size}_{CFG.NUM_LAYERS}_{CFG.HIDDEN_SIZE}.png")

    def predict(self, test_dataset):

        test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)

        # load best weight
        state_dict = load_weight(f"result/{self.window_size}_{CFG.NUM_LAYERS}_{CFG.HIDDEN_SIZE}.pth")
        self.model.load_state_dict(state_dict['state'])
        self.model = self.model.to(self.device)

        timestamps, distance, attacks = self._predict(test_loader)
        anomaly_score = np.mean(distance, axis=1)
        c_anomaly_score = range_check(anomaly_score, 60)
        labels = put_labels(c_anomaly_score, state_dict['threshold'])

        submission = pd.read_csv('/home/salmon21/coco/Competition/HAICON2021/sample_submission.csv')
        submission.index = submission['timestamp']
        submission.loc[timestamps, 'attack'] = labels

        submission.to_csv(f"result/{self.window_size}_{CFG.NUM_LAYERS}_{CFG.HIDDEN_SIZE}_submission.csv", index=False)

import os
import argparse
import random
import math

import numpy as np
import pandas as pd
import mne
from mne import read_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

import torch
from torch.utils.data import Dataset

import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import MySeq2SeqFastAttention

np.set_printoptions(suppress=True)


MAX_LEN = 5000
# PAD_TOKEN = -1

device = torch.device('cpu')

SOS_TOKEN = -1

anno_dict = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": 6,
    "Movement time": 7
}

num_annos = len(np.unique(list(anno_dict.values())))

parser = argparse.ArgumentParser(description='Train the Neural Network to process sleep data')
parser.add_argument('--root_dir', metavar='R', dest='root_dir', type=str, default='pandas_seq',
                    help='Root directory for the dataset')
parser.add_argument('--batch', metavar='B', dest='batch_size', type=int, default=10,
                    help='Batch Size')
parser.add_argument('--num_workers', metavar='N', dest='n_workers', type=int, default=4,
                    help='Number of Workers')
parser.add_argument('--seed', metavar='S', dest='seed', default=2020,
                    help='Pseudorandom seed')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
parser.add_argument('--optimizer', metavar='O', type=str, default='adam', help='Optimizer')
args = parser.parse_args()

class SleepDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args: TBD
        """
        self.root_dir = root_dir
        record = f'{root_dir}/RECORDS'
        self.fnames = pd.read_csv(record, header=None)


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # return self.X_dat[idx], self.y_dat[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print(self.fnames[0].iloc[idx])

        edf_name = os.path.join(self.root_dir, self.fnames[0].iloc[idx])
        
        edf_dir, patient_night = edf_name.rsplit('/', 1)
        patient_night = patient_night[:6]
        
        hyp_name = None
        for f in os.listdir(edf_dir):
            if patient_night in f and 'Hypnogram' in f:
                hyp_name = os.path.join(edf_dir, f)
                break

        if hyp_name is None:
            raise Exception('Could not find annotations...')

        psg = read_raw_edf(edf_name, preload=True)
        annotations = read_annotations(hyp_name)

        sfreq = int(psg.info['sfreq'])
        psg_df = psg.to_data_frame().drop(columns='time')

        y = np.zeros(len(psg_df), dtype=int)

        # print(annotations)
        for a in annotations:
            a_onset = a['onset']
            a_duration = a['duration']
            a_label = a['description']
            
            i = int(a_onset * sfreq)
            j = int((a_onset + a_duration) * sfreq)

            y[i:j] = anno_dict[a_label]

        return psg_df, y

def train(model, optimizer, loss_criterion, sd, train_ind, e0 = 0, epochs=20, verbose=False):

    start = time.time()
    for epoch in range(e0, e0+epochs): # epochs
        losses = []
        
        np.random.shuffle(train_ind)

        epoch_start = time.time()
        for i in train_ind:
            psg_df, annotations = sd[i]

            # seq_start = time.time()
            # for j in range(0, len(psg_df), MAX_LEN):

            j = random.randint(0, len(psg_df))

            # if verbose:
            #     print(f'Starting minibatch {int(j/MAX_LEN)+1}/{int(math.floor(len(psg_df) / MAX_LEN))}')

            psg_df_j = psg_df.iloc[j:j+MAX_LEN]
            anno_j = annotations[j:j+MAX_LEN]

            # psg_df_j = (psg_df - psg_df.min()) / (psg_df.max() - psg_df.min())
            # anno_j[0] = anno_dict['SOS']
            # anno_j[-1] = anno_dict['EOS']

            # print(psg_df_j)
            # print(anno_j)

            eeg = psg_df_j['EEG Fpz-Cz'].to_numpy().reshape(-1,1)
            anno_j = anno_j.reshape(-1,1)
            eeg = torch.from_numpy(eeg).float().to(device)

            anno_1h = np.zeros((anno_j.shape[0], num_annos))
            anno_1h[np.arange(anno_j.shape[0]), anno_j] = 1
            anno_1h = torch.from_numpy(anno_1h).float().to(device)
            anno_j = torch.from_numpy(anno_j).long().to(device)

            out = model(eeg, anno_1h)
            out = out.contiguous().view(-1, num_annos)

            loss = loss_criterion(out, anno_j.view(-1))

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            d_seq_t = time.time() - epoch_start
            # seq_left = d_seq_t / (j + MAX_LEN) * (len(psg_df) - (j + MAX_LEN))
            seq_left = d_seq_t / len(losses) * (len(train_ind) - len(losses))

            if verbose:
                print(f'Epoch: {epoch} ({len(losses)}/{len(train_ind)})\tLoss: {loss} \tEpoch Time {d_seq_t} \tEpoch Left {seq_left}')

        torch.save(model.state_dict(), f'model_epoch_{epoch}.dict')
        np.save(f'loss_epoch_{epoch}.npy', losses)
    

def test(model, sd, test_ind, num_tests, verbose=False):
    start = time.time()

    correct = 0
    total = 0

    confusion = np.zeros((num_annos, num_annos))

    with torch.no_grad():

        for i in random.choices(test_ind, k=num_tests):
            psg_df, annotations = sd[i]
            j = random.randint(0, len(psg_df))

        # for i in test_ind:
        #     psg_df, annotations = sd[i]

            # for j in range(0, len(psg_df), MAX_LEN * 5):
            psg_df_j = psg_df.iloc[j:j+MAX_LEN]
            anno_j = annotations[j:j+MAX_LEN]

            # psg_df_j = (psg_df_j - psg_df.min()) / (psg_df.max() - psg_df.min())

            eeg = psg_df_j['EEG Fpz-Cz'].to_numpy().reshape(-1,1)
            anno_j = anno_j.reshape(-1,1)
            eeg = torch.from_numpy(eeg).float().to(device)

            anno_1h = np.zeros((anno_j.shape[0], num_annos))
            anno_1h[np.arange(anno_j.shape[0]), anno_j] = 1
            anno_1h = torch.from_numpy(anno_1h).float().to(device)
            anno_j = torch.from_numpy(anno_j).long().to(device)

            input_trg = np.ones(anno_1h.shape) * SOS_TOKEN
            input_trg = torch.from_numpy(input_trg).float().to(device)

            output_trg = model(eeg, input_trg)
            
            _,output = torch.max(output_trg, 2)

            output = output.flatten().numpy()
            anno_j = anno_j.flatten().numpy()
            
            # print(output_trg)
            # print(output)
            # print(anno_j)

            if torch.isnan(output_trg).any():
                raise Exception('Found NaN in NN output.')

            correct += np.sum(output== anno_j)
            total += len(anno_j)

            for x,y in zip(output, anno_j):
                confusion[x,y] += 1

            dt = time.time() - start
            print(f'Accuracy: {correct/total*100}% \tTested {total} \t ({i+1}/{len(train_ind)}) \tTime elapsed: {dt}')
            print('Confusion Matrix:')
            print(confusion)

def dataset_stats(dataset):
    train_dist = np.zeros(num_annos)
    test_dist = np.zeros(num_annos)
    distribution = np.zeros(num_annos)
    for idx, (psg_df, annotations) in enumerate(sd):
        unique, counts = np.unique(annotations, return_counts=True)
        for i,v in zip(unique, counts):
            distribution[i] += v
            if idx < 100:
                train_dist[i] += v
            else:
                test_dist[i] += v
    
    print(anno_dict)
    print('---totals---')
    print(distribution)
    print(train_dist)
    print(test_dist)
    
    distribution *= 100 / np.sum(distribution)
    train_dist *= 100 / np.sum(train_dist)
    test_dist *= 100 / np.sum(test_dist)

    print('---percent---')
    print(distribution)
    print(train_dist)
    print(test_dist)

if __name__ == '__main__':
    root_dir = 'sleep-edf-database-expanded-1.0.0'
    filename = f'{root_dir}/RECORDS'
    learning_rate = args.lr
    verbose = True

    sd = SleepDataset(root_dir)
    print(f'Dataset size: {len(sd)}')

    # dataset_stats(sd)
    
    model = MySeq2SeqFastAttention(num_annos, 100, 100, 128, 10)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Learning method not recommend for task")

    loss_criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('model_epoch_19.dict'))

    train_ind = np.arange(100)
    train(model, optimizer, loss_criterion, sd, train_ind, e0 = 20, epochs=20, verbose=verbose)

    test_ind = np.arange(100, len(sd))
    num_tests = 1000
    test(model, sd, test_ind, num_tests, verbose)


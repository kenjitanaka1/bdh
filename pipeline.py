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
# Limits the amount of scientific notation that numpy uses.
np.set_printoptions(suppress=True)

# Sets the maximum sequence length used for testing and training
MAX_LEN = 5000

# Forces cpu since I don't have an nvidia graphics card
device = torch.device('cpu')

# dictionary mapping sleep annotations to numbers
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
# gets the number of annotations used
num_annos = len(np.unique(list(anno_dict.values())))
# parse input arguments: dataset directory, epochs, 
parser = argparse.ArgumentParser(description='Train the Neural Network to process sleep data')
parser.add_argument('--root_dir', metavar='R', dest='root_dir', type=str, default='pandas_seq',
                    help='Root directory for the dataset')
parser.add_argument('--seed', metavar='S', dest='seed', default=2020,
                    help='Pseudorandom seed')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
parser.add_argument('--optimizer', metavar='O', type=str, default='adam', help='Optimizer')
args = parser.parse_args()

class SleepDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Initializes a sleep dataset based on a root directory. 
        Based on the sleep edfx file structure.
        Just holds the file names of all the files to start
        """
        self.root_dir = root_dir
        record = f'{root_dir}/RECORDS'
        self.fnames = pd.read_csv(record, header=None)


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        """
        When an item is requested, loads the edf into a pandas dataframe, 
        finds the corresponding annotations
        and converts the annotations list to a numpy array 
        of the classifations the same length as the edf. 

        Returns the ith edf according to RECORDS and the corresponding annotations
        """
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
    """
    Trains a given model with a given optimizer and loss criterion
    sd is the sleepDataset as specified above
    train_ind is the indicies that the model should train on
    e0 is the epoch to start with. Useful if you have already trained the model for a while and want to continue training
    epochs = number of epochs to train the model for
    verbose = enable some printouts 
    """
    start = time.time()
    for epoch in range(e0, e0+epochs): # epochs
        losses = []
        # basically just picks a random recording to choose a sample from
        np.random.shuffle(train_ind)
        # recall the starting time for statistics purposes
        epoch_start = time.time()
        for i in train_ind:
            # load file, choose some subset of the recording to train on
            psg_df, annotations = sd[i]

            j = random.randint(0, len(psg_df))


            psg_df_j = psg_df.iloc[j:j+MAX_LEN]
            anno_j = annotations[j:j+MAX_LEN]
            # choose just the EEG Fpz-Cz signal
            eeg = psg_df_j['EEG Fpz-Cz'].to_numpy().reshape(-1,1)
            anno_j = anno_j.reshape(-1,1)
            eeg = torch.from_numpy(eeg).float().to(device)
            # reshapes annotations and makes it 1 hot, for input to the model
            anno_1h = np.zeros((anno_j.shape[0], num_annos))
            anno_1h[np.arange(anno_j.shape[0]), anno_j] = 1
            anno_1h = torch.from_numpy(anno_1h).float().to(device)
            anno_j = torch.from_numpy(anno_j).long().to(device)
            # gets an output from the model, reshapes it
            out = model(eeg, anno_1h)
            out = out.contiguous().view(-1, num_annos)
            # calculate loss
            loss = loss_criterion(out, anno_j.view(-1))
            # step the optimizer
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # record the time it took for this part of the training
            d_seq_t = time.time() - epoch_start
            seq_left = d_seq_t / len(losses) * (len(train_ind) - len(losses))
            # print out for diagnosis
            if verbose:
                print(f'Epoch: {epoch} ({len(losses)}/{len(train_ind)})\tLoss: {loss} \tEpoch Time {d_seq_t} \tEpoch Left {seq_left}')
        # save the model and loss every epoch
        torch.save(model.state_dict(), f'model_with_linear/model_epoch_{epoch}.dict')
        np.save(f'model_with_linear/loss_epoch_{epoch}.npy', losses)
    

def test(model, sd, test_ind, num_tests, verbose=False):
    """
    Tests a given model
    sd is the sleep dataset
    test_ind is the indicies in the sleep dataset to use to test on
    num_tests is how many test to run 
    verbose is if printouts should happen or not. Will print out accuracy so far and a confusion matrix if verbose=True
    """
    start = time.time()
    # keep tally of total and correct 
    correct = 0
    total = 0
    # keep a confusion matrix
    confusion = np.zeros((num_annos, num_annos))

    with torch.no_grad():
        # generates some ordering of random files to choose samples from
        for i in random.choices(test_ind, k=num_tests):
            # load file, pick a sample from the recording. Same as training
            psg_df, annotations = sd[i]
            j = random.randint(0, len(psg_df))

            psg_df_j = psg_df.iloc[j:j+MAX_LEN]
            anno_j = annotations[j:j+MAX_LEN]
            # pick only EEG Fpz-Cz channel, reshape
            eeg = psg_df_j['EEG Fpz-Cz'].to_numpy().reshape(-1,1)
            anno_j = anno_j.reshape(-1,1)
            eeg = torch.from_numpy(eeg).float().to(device)
            # reshape annotations and make it one hot
            anno_1h = np.zeros((anno_j.shape[0], num_annos))
            anno_1h[np.arange(anno_j.shape[0]), anno_j] = 1
            anno_1h = torch.from_numpy(anno_1h).float().to(device)
            anno_j = torch.from_numpy(anno_j).long().to(device)

            # create input of 1's to feed into the network. Seems to work better than 0's
            input_trg = np.ones(anno_1h.shape) * anno_dict['Sleep stage W']
            input_trg = torch.from_numpy(input_trg).float().to(device)
            # run network
            output_trg = model(eeg, input_trg)
            # convert to classes based on highest probability
            _,output = torch.max(output_trg, 2)
            # compare
            output = output.flatten().numpy()
            anno_j = anno_j.flatten().numpy()
            # NaN check to warn about errors caused from learning rate being to high
            if torch.isnan(output_trg).any():
                raise Exception('Found NaN in NN output.')
            # counts correct
            correct += np.sum(output== anno_j)
            total += len(anno_j)
            # record data into confusion matrix
            for x,y in zip(output, anno_j):
                confusion[x,y] += 1
            
            dt = time.time() - start
            if verbose:
                print(f'Accuracy: {correct/total*100}% \tTested {total} \t ({i+1}/{len(test_ind)}) \tTime elapsed: {dt}')
                print('Confusion Matrix:')
                print(confusion)

    return correct/total

def dataset_stats(dataset):
    """
    Generates printouts talking about the distribution of the training, testing, and overall dataset
    """
    # just count up for each occurence of an annotation
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
    # print out actual values
    print(anno_dict)
    print('---totals---')
    print(distribution)
    print(train_dist)
    print(test_dist)
    # print out percentages
    distribution *= 100 / np.sum(distribution)
    train_dist *= 100 / np.sum(train_dist)
    test_dist *= 100 / np.sum(test_dist)

    print('---percent---')
    print(distribution)
    print(train_dist)
    print(test_dist)

if __name__ == '__main__':
    # directories, verbose flag
    root_dir = 'sleep-edf-database-expanded-1.0.0'
    filename = f'{root_dir}/RECORDS'
    learning_rate = args.lr
    verbose = True
    # create dataset
    sd = SleepDataset(root_dir)
    print(f'Dataset size: {len(sd)}')

    # print datset stats if this isn't commented out
    # dataset_stats(sd)
    # create model
    model = MySeq2SeqFastAttention(num_annos, 100, 100, 128, 10)
    # make optimizer based on args
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Learning method not recommend for task")
    # loss criterion is cross entropy 
    loss_criterion = nn.CrossEntropyLoss()
    # load model state, can comment out train and just test
    # model.load_state_dict(torch.load('model_epoch_21.dict'))
    # train the model for a number of epochs
    train_ind = np.arange(100)
    train(model, optimizer, loss_criterion, sd, train_ind, e0 = 0, epochs=args.epochs, verbose=verbose)
    # then test 1000 segments
    test_ind = np.arange(100, len(sd))
    num_tests = 1000
    test(model, sd, test_ind, num_tests, verbose)


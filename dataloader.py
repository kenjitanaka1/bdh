import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne import read_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

import torch
from torch.utils.data import Dataset

MAX_LEN = 6000

device = torch.device('cpu')

anno_dict = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

dict_anno = {
    0: 'Sleep stage W',
    1: 'Sleep stage 1',
    2: 'Sleep stage 2',
    3: 'Sleep stage 3 or 4',
    4: 'Sleep stage R',
    5: 'Sleep stage ? or Movement time'
}

num_annos = len(np.unique(list(anno_dict.values())))

class SleepDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args: TBD
        """
        fnames = os.listdir(root_dir)[:100]#[:10000]
        self.X_dat = []
        self.y_dat = []

        for f in fnames:
            f = os.path.join(root_dir, f)

            with np.load(f, allow_pickle=True) as data:
                Y = data['y']
                X = data['x']
                # fs = data['fs']

                X = torch.from_numpy(X).float().to(device)
                Y = torch.from_numpy(Y).long().reshape(-1,1).to(device)

                self.X_dat.append(X)
                self.y_dat.append(Y)


    def __len__(self):
        return len(self.y_dat)

    def __getitem__(self, idx):
        return self.X_dat[idx], self.y_dat[idx]
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # f = os.path.join(self.root_dir, self.fnames[idx])

        # with np.load(f, allow_pickle=True) as data:
        #     y = data['y']
        #     X = data['x']
        #     fs = data['fs']

        # temp = np.zeros(max_shape)
        # temp[:X.shape[0],:X.shape[1],:X.shape[2]] = X
        # X = temp

        # print(X.shape)
        
        # if self.transform:
        #     X, y = self.transform(X, y)

        # return X, y, fs

def processDataset(root_dir, out_dir, tstep=30):
    """
    inputs:
        tstep - time step (seconds)
    """
    record = f'{root_dir}/RECORDS'
    fnames = pd.read_csv(record, header=None)

    tstep = int(round(tstep))

    os.makedirs(out_dir, exist_ok=True)

    max_seq_len = -1

    for _, fname in fnames.iterrows():
        edf_name = os.path.join(root_dir, fname[0])
        
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
        istep = sfreq * tstep
        psg_df = psg.to_data_frame().drop(columns='time')

        if len(psg_df.shape) is not 2 or psg_df.shape[1] is not 7:
            continue

        X = []
        y = []

        for i in range(0, len(psg_df), istep):
            t = int(i / sfreq)
            tnext = t + tstep

            if tnext < len(psg_df):
                psg_t = psg_df.iloc[i:i + istep]
                a_t = None
                for a in annotations:
                    a_onset = a['onset']
                    a_duration = a['duration']
                    if a_onset <= t and tnext <= a_onset + a_duration:
                        a_t = a['description']
                        break
                if a_t is not None:
                    X.append(psg_t.to_numpy())
                    y.append(anno_dict[a_t])

        X = np.array(X)
        y = np.array(y)

        if len(X.shape) < 3:
            continue

        save_dict = {
            'x': X,
            'y': y,
            'fs': sfreq
        }

        if len(X) > max_seq_len:
            max_seq_len = len(X)

        file_out = os.path.join(out_dir, f"{patient_night[:6]}.npz")
        np.savez(file_out, **save_dict)
    
    print(f'Maximum sequence length: {max_seq_len}')

SAMPLE_LEN = 60 #seconds

def preprocess(root_dir, out_dir, tstep = 30):
    record = f'{root_dir}/RECORDS'
    fnames = pd.read_csv(record, header=None)

    # tstep = int(round(tstep))

    os.makedirs(out_dir, exist_ok=True)
    
    # Takes 1 minute sequences out of the datasets, 1 of each static sequence, and 1 of each transition

    # X_samp = []
    # y_samp = []

    for _k, fname in fnames.iterrows():
        print(f'starting iter {_k}/{len(fnames)}')
        edf_name = os.path.join(root_dir, fname[0])
        
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

        if len(psg_df.shape) is not 2 or psg_df.shape[1] is not 7:
            continue

        for i, anno in enumerate(annotations):
            a_onset = anno['onset']
            a_duration = anno['duration']

            if a_duration >= SAMPLE_LEN:
                #sample
                start = int((a_onset + a_duration / 2 - SAMPLE_LEN / 2) * sfreq)
                end = int((a_onset + a_duration / 2 + SAMPLE_LEN / 2) * sfreq)
                const_samp = psg_df.iloc[start:end].copy()
                const_anno = np.ones(sfreq * SAMPLE_LEN) * anno_dict[anno['description']]

                if len(const_samp) == len(const_anno) and len(const_samp) > 0:
                    save_dict = {
                        'x': const_samp,
                        'y': const_anno,
                        'fs': sfreq
                    }
                    file_out = os.path.join(out_dir, f"{patient_night[:6]}_{i}_c.npz")
                    np.savez(file_out, **save_dict)

            a_next = annotations[i + 1] if i + 1 < len(annotations) else None

            if a_duration >= SAMPLE_LEN / 2 and a_next is not None and a_next['duration'] > SAMPLE_LEN / 2:
                start = int((a_onset + a_duration - SAMPLE_LEN / 2) * sfreq)
                end = int((a_onset + a_duration + SAMPLE_LEN / 2) * sfreq)
                transition_samp = psg_df.iloc[start:end].copy()
                transition_anno = np.zeros(sfreq * SAMPLE_LEN)
                mid = int(sfreq * SAMPLE_LEN / 2)
                transition_anno[:mid] = anno_dict[anno['description']]
                transition_anno[mid:] = anno_dict[a_next['description']]
                
                if len(transition_samp) == len(transition_anno) and len(transition_samp) > 0:
                    save_dict = {
                        'x': transition_samp,
                        'y': transition_anno,
                        'fs': sfreq
                    }
                    file_out = os.path.join(out_dir, f"{patient_night[:6]}_{i}_t.npz")
                    np.savez(file_out, **save_dict)
                
            # X_samp.append(const_samp)
            # y_samp.append(const_anno)
            # X_samp.append(transition_samp)
            # y_samp.append(transition_anno)

        print(f'finished iter {_k + 1}/{len(fnames)}')

    # save_dict = {
    #     'x': X_samp,
    #     'y': y_samp,
    # }

    # file_out = os.path.join(out_dir, f"data.npz")
    # np.savez(file_out, save_dict)


if __name__ == '__main__':
    root_dir = 'sleep-edf-database-expanded-1.0.0'
    filename = f'{root_dir}/RECORDS'
    out_dir = 'pandas_seq'

    # print('Processing Dataset...')
    # # processDataset(root_dir, out_dir)
    # preprocess(root_dir, out_dir)
    # print('Done Processing.')

    print('Testing loading dataset...')
    sd = SleepDataset(out_dir)
    for i in range(0,len(sd)):
        X,_=sd[i]
        print(X.shape)
    print('Success loading dataset!')
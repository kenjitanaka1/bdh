import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import random
import math
import numpy as np

from dataloader import SleepDataset, anno_dict, dict_anno, MAX_LEN as MAX_LENGTH
from model import EncoderRNN, AttnDecoderRNN

SOS_token = 0
EOS_token = -1

device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Train the Neural Network to process sleep data')
parser.add_argument('--root_dir', metavar='R', dest='root_dir', type=str, default='pandas_seq',
                    help='Root directory for the dataset')
# parser.add_argument('--batch', metavar='B', dest='batch_size', type=int, default=10,
#                     help='Batch Size')
parser.add_argument('--num_workers', metavar='N', dest='n_workers', type=int, default=4,
                    help='Number of Workers')
parser.add_argument('--seed', metavar='S', dest='seed', default=2020,
                    help='Pseudorandom seed')
# parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')
args = parser.parse_args()

def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        # input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, dataloader, n=10):
    acc = 0
    total = 0

    for _, (data, target) in enumerate(dataloader):
        for i in range(data.shape[0]):
            in_tensor = data[i,:,:]
            out_tensor = target[i,:]
            print('>', in_tensor)
            print('=', out_tensor)
            output_seq, attentions = evaluate(encoder, decoder, in_tensor)
            # output_sentence = ' '.join(output_seq)
            print('<', np.array(output_seq))
            print('')
            acc += np.sum(out_tensor.numpy().flatten() == np.array(output_seq).flatten())
            print((out_tensor.numpy() == np.array(output_seq)).shape)
            print(out_tensor.shape)
            total += np.product(out_tensor.shape)

    print(f'Test Accuracy: {acc/total}')
    

if __name__ == '__main__':

    sleepDataset = SleepDataset(args.root_dir)

    val_ind = range(1,10000, 75 * 3)
    valid_sampler = SubsetRandomSampler(val_ind)
    test_dataloader = DataLoader(sleepDataset,
                        num_workers=args.n_workers, sampler=valid_sampler)
 
    max_seq_len = 6000 # hardcoded
    hidden_size = 512
    in_size = 7
    num_annos = 6
    
    encoder = EncoderRNN(in_size, hidden_size, device).to(device)
    decoder = AttnDecoderRNN(hidden_size, num_annos, device, max_seq_len, dropout_p=0.1).to(device)    
    
    encoder.load_state_dict(torch.load('encoder.dict'))
    decoder.load_state_dict(torch.load('decoder.dict'))

    evaluateRandomly(encoder, decoder, test_dataloader)

    # output_words, attentions = evaluate(
    #     encoder, decoder, "je suis trop froid .")
    # plt.matshow(attentions.numpy())
    # plt.show()
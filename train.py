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
from model import EncoderRNN, AttnDecoderRNN, MySeq2SeqAttention

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
args = parser.parse_args()

device = torch.device('cpu')

SOS_token = 0
EOS_token = -1
teacher_forcing_ratio = 0.5

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device).float()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def train_iters(encoder, decoder, dataloader, learning_rate=1e-3, print_every=1000, plot_every=100):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for batch_idx, (data,target) in enumerate(dataloader):
        print(f'Starting epoch {batch_idx+1}/{len(dataloader)}')
        for i in range(data.shape[0]):
            in_tensor = data[i,:,:]
            out_tensor = target[i,:]
            loss = train(in_tensor, out_tensor, encoder,
                decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            iter = i + 1

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(dataloader)),
                                            iter, iter / len(dataloader) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

if __name__ == '__main__':

    sleepDataset = SleepDataset(args.root_dir)

    print(len(sleepDataset))

    train_ind = range(0,10000, 65)
    train_sampler = SubsetRandomSampler(train_ind)
    dataloader = DataLoader(sleepDataset, batch_size=args.batch_size,
                        num_workers=args.n_workers, sampler=train_sampler)

    max_seq_len = 6000 # hardcoded
    hidden_size = 512
    in_size = 7
    num_annos = 6
    # encoder1 = EncoderRNN(in_size, hidden_size, device).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, num_annos, device, max_seq_len, dropout_p=0.1).to(device)    
    
    # train_iters(encoder1, attn_decoder1, dataloader, print_every=1)

    # torch.save(encoder1.state_dict(), 'encoder.dict')
    # torch.save(attn_decoder1.state_dict(), 'decoder.dict')

    model = MySeq2SeqAttention(in_size, hidden_size, 
                               num_annos, hidden_size)

    
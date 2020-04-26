"""
This file exists to create the loss and accuracy graphs for the report. 
"""
import os

import torch
import numpy as np
from matplotlib import pyplot as plt

from model import MySeq2SeqFastAttention
from pipeline import test, num_annos, SleepDataset

# This tests the accuracy of the model I trained. 
# It's hardcoded to just do 22 epochs, skipping the odd ones to save time. 
def test_accs():
    # create dataset
    root_dir = 'sleep-edf-database-expanded-1.0.0'
    sd = SleepDataset(root_dir)

    # create model and acces folder where models are stored. 
    thedir = 'model_with_linear'
    model = MySeq2SeqFastAttention(num_annos, 100, 100, 128, 10)
    # test indicies 100:197
    test_ind = np.arange(100, len(sd))
    # how many tests to run
    num_tests = 25
    accs = []
    for i in range(0,22,2):
        # load model
        path = os.path.join(thedir, f'model_epoch_{i}.dict')
        model.load_state_dict(torch.load(path))
        # test model
        acc_i = test(model, sd, test_ind, num_tests)
        accs.append(acc_i)

    return accs

if __name__ == '__main__':
    thedir = 'model_with_linear'
    ## This will generate the accuracy file, using the above function. 
    ## Takes a while so it's commented out. 
    # accs = test_accs()
    # np.save('accuracies.npy',accs)

    avg_losses = []
    # load all of the losses per epoch and take the average
    for i in range(22):
        path = os.path.join(thedir, f'loss_epoch_{i}.npy')
        losses = np.load(path)
        avg_losses.append(np.mean(losses))


    print(avg_losses)
    print(accs)
    
    # plot avg loss per epoch
    plt.plot(avg_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss Per Epoch')
    plt.savefig('loss_graph.png')
    plt.show()

    # plot avg accuracy per epoch
    accs = np.load('accuracies.npy')
    plt.figure()
    plt.plot(accs)
    plt.xticks(range(0,22,2))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Per Epoch')
    plt.savefig('accuracy_graph.png')
    plt.show()
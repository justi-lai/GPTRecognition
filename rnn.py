import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

class RNN(nn.Module):
    def __init__(self, , hidden_size, num_layers, output_dim, embedding_size, vocab_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, nonlinearity="relu")
        self.embedding = torch.Embedding(vocab_size, embedding_size)
        self.W = nn.Linear(hidden_size, output_dim)
        self.softmax = nn.Softmax()
        self.loss = nn.NLLLoss()

    def compute_loss(self, predicted_output, golden_label):
        return self.loss(predicted_output, golden_label)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        out, hn = self.rnn(embedding)
        results = self.W(out[-1])
        results = self.softmax(results)
        return results

def load_data(file_name):
    #fill this eventually
    return train_data, valid_data, vocab_size

print("===== Loading Data =========")
train_data, valid_data, vocab_size = load_data()

print("===== Vectorizing Data =====")
embedding_size = 128
vocab_size = vocab_size
hidden_size = 256
num_layers = 2
output_dim = 2

model = RNN(hidden_size, num_layers, output_dim, embedding_size, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epoch = 0
stopping_condition = False

while not stopping_condition:
    random.shuffle(train_data)
    model.train()
    print("Started training for epoch: {}".format(epoch + 1))

    minibatch_size = 16
    N = len(train_data)
    correct = 0
    total = 0

    for minibatch_index in tqdm(N//minibatch_size):
        optimizer.zero_grad()
        loss = None
        for example_index in range(minibatch_size):
            input_indices, golden_label = train_data[minibatch_index * minibatch_size + example_index]
            input_indices = torch.tensor(input_indices).view(-1, 1)


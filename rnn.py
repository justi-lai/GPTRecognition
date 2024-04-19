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
    def __init__(self, input_dim, h, layer_dim, output_dim, embedding_size):
        super(RNN, self).__init__()
        self.hidden_dim = h
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim + embedding_size, h, layer_dim, nonlinearity="relu")
        self.W = nn.Linear(h, output_dim)
        self.softmax = nn.Softmax()
        self.loss = nn.NLLLoss()

    def computeLoss(self, predicted_output, golden_label):
        return self.loss(predicted_output, golden_label)

    def forward(self, inputs):
        out, hn = self.rnn(inputs)
        results = self.W(out[-1])
        results = self.softmax(results)
        return results


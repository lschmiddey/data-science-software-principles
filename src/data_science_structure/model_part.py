from config import *

import torch
from torch import nn


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


def conv1d(ni, nf, ks, stride):
    return nn.Sequential(
        nn.Conv1d(ni, nf, ks, stride=stride, padding=0), nn.BatchNorm1d(nf), nn.ReLU())


def get_cnn_layers(input_shape, output_shapes:list, kernels:list, strides:list, drop=.5):
    output_shapes = [input_shape] + output_shapes
    return [
        conv1d(output_shapes[i], output_shapes[i+1], kernels[i], strides[i])
        for i in range(len(output_shapes)-1)
    ] + [nn.MaxPool1d(2, stride=2), Flatten(), nn.Dropout(drop), nn.Linear(output_shapes[-1], 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(64, 64), nn.ReLU(inplace=True)]


class Classifier(nn.Module):
    """Model Baseclass."""

    def __init__(self, conv_layers, emb_dims, no):
        super().__init__()

        self.raw = conv_layers

        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.emb_dims = emb_dims

        self.emb_out = nn.Sequential(
            nn.Linear(no_of_embs, 64), nn.ReLU(inplace=True), nn.Linear(64, 64))

        self.out = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

    def forward(self, t_raw, embeddings):  # this is where the data flows in later in the training
        raw_out = self.raw(t_raw)
        emb = [emb_layer(embeddings[:, i].long()) for i, emb_layer in enumerate(self.embeddings)]
        # we want to concatenate convolutions and embeddings. Embeddings are of size (batch_size, no_of_embs),
        # convolution of size (batch, 256, 1) so we need to add another dimension to the embeddings at dimension 2 (
        # counting starts from 0)
        emb_cat = torch.cat(emb, 1)
        emb_cat = self.emb_out(emb_cat)
        t_in = torch.cat([raw_out, emb_cat], dim=1)
        out = self.out(t_in)
        return out
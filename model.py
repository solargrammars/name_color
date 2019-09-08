import numpy as np
import torch
import torch.nn as nn
import ipdb

class NameColor(nn.Module):
    def __init__(self, 
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            output_dim, 
            n_layers,  
            dropout, 
            pad_idx,
            return_hidden=False):

        super().__init__()

        self.embedding = nn.Embedding(
                vocab_size, 
                embedding_dim, 
                padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_dim,
                num_layers=n_layers, 
                bidirectional=True,
                dropout=dropout,
                batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.return_hidden = return_hidden

    def forward(self, text, text_lengths):
        
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths, batch_first= True )
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first = True)
        
        hidden = self.dropout(
                torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        if self.return_hidden:
            return (self.fc(hidden.squeeze(0)), hidden)
        else:
            return self.fc(hidden.squeeze(0)) 
                
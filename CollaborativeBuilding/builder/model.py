import math
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from builder.dataloader_with_glove import BuilderDataset, RawInputs

sys.path.append('..')
from utils import *

color2id = {
    "orange": 0,
    "red": 1,
    "green": 2,
    "blue": 3,
    "purple": 4,
    "yellow": 5
}
id2color = {v:k for k, v in color2id.items()}


class Builder(nn.Module):
    def __init__(self, config, vocabulary):
        super(Builder, self).__init__()
        self.encoder = UtteranceEncoder(config["encoder_config"], vocabulary)
        self.decoder = ActionDecoder(config['decoder_config'])

    def forward(self, encoder_inputs, grid_repr_inputs, action_repr_inputs, label, location_mask=None, raw_input=None, dataset=None):
        dialogue_repr = self.encoder(encoder_inputs)
        loss, acc, predicted_seq = self.decoder(dialogue_repr, grid_repr_inputs, action_repr_inputs, label, location_mask, raw_input, dataset)
        return loss, acc, predicted_seq


class UtteranceEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super(UtteranceEncoder, self).__init__()
        self.rnn_hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.embed_dropout = config['embed_dropout']
        self.bidirectional = config['bidirectional']
        self.rnn_dropout = config['rnn_dropout']
        self.train_embeddings = config['train_embeddings']
        self.mlp_output = config['color_size']
        self.mlp_dropout = config['mlp_dropout']
        self.mlp_input = self.rnn_hidden_size
        self.mlp_hidden = self.rnn_hidden_size
        self.word_embedding_size = vocabulary.embed_size

        self.embed_mapping = vocabulary.word_embeddings
        if self.train_embeddings: 
            self.embed_mapping.weight.requires_grad = True
        self.embed_dropout = nn.Dropout(p=self.embed_dropout)
        self.rnn = nn.GRU(self.word_embedding_size, self.rnn_hidden_size, self.num_hidden_layers, dropout=self.rnn_dropout, bidirectional=self.bidirectional, batch_first=True)
        self.init_weights()

    def forward(self, encoder_inputs):
        """
        encoder_inputs: [batch_size, seq_length], where seq_length is variable here
        """
        embedded = self.embed_mapping(encoder_inputs) # [batch_size, seq_length, hidden_size], where hidden_size = 300
        batch_size, seq_len, _ = embedded.shape
        embedded = self.embed_dropout(embedded) 
        packed = pack_padded_sequence(embedded, [seq_len]*batch_size, batch_first=True)
        output_packed, hidden = self.rnn(packed) 
        output_unpacked, lens_unpacked = pad_packed_sequence(output_packed, batch_first=True)
        output = output_unpacked.view(batch_size, seq_len, 2 if self.bidirectional else 1 , self.rnn_hidden_size)
        output = torch.mean(output, dim=-2)
        return output # [batch_size, max_length, 300]

    def init_weights(self):
        """ Initializes weights of linear layers with Xavier initialization. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)

    def take_last_hidden(self, hidden, num_hidden_layers, bidirectional, batch_size, rnn_hidden_size):
        hidden = hidden.view(num_hidden_layers, bidirectional, batch_size, rnn_hidden_size) # (num_layers, num_directions, batch, hidden_size)
        hidden = hidden[-1] # hidden: (num_directions, batch, hidden_size)
        return hidden


class WorldStateEncoderCNN(nn.Module):
    def __init__(self, config):
        super(WorldStateEncoderCNN, self).__init__()
        self.num_conv_layers = config['num_conv_layers']
        self.world_dim = config['world_dim']
        self.kernel_size = config['kernel_size']
        self.input_size = config['world_hidden_size']

        self.conv_layers = []
        for i in range(self.num_conv_layers):
            if i == 0:
                layer = nn.Conv3d(self.world_dim, self.input_size, kernel_size=self.kernel_size, stride=1, padding=1)
            else:
                layer = nn.Conv3d(self.input_size, int(self.input_size/2), kernel_size=1, stride=1, padding=0)
                self.conv_layers.append(layer)
                layer = nn.Conv3d(int(self.input_size/2), self.input_size, kernel_size=self.kernel_size, stride=1, padding=1)
            self.conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.nonlinearity = nn.ReLU()
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, world_repr):
        """
        input: [1, 8, 11, 9, 11]
        output: [1, 300, 11, 9, 11]
        """
        for i, conv in enumerate(self.conv_layers):
            world_repr = self.dropout(self.nonlinearity(conv(world_repr)))
        return world_repr.permute(0, 2, 3, 4, 1).view(-1, 1089, self.input_size)


class ActionDecoder(nn.Module):
    def __init__(self, config):
        super(ActionDecoder, self).__init__()
        self.text_hidden_size = config['text_hidden_size']
        self.text_dropout = config['text_dropout']
        self.text_attn_heads = config['text_attn_heads']
        self.text_layers = config['text_layers']

        self.world_hidden_size = c
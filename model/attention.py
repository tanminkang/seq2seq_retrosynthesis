import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import math


class EncoderRNN(nn.Module):
    def __init__(self, voc, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.voc = voc
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(voc.size, embed_size)
        self.gru1 = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.gru3 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, h=None):
        output = self.embed(input)
        output, _ = self.gru1(output, None)
        output, _ = self.gru2(output, None)
        output, h_out = self.gru3(output, None)
        return output, torch.cat([h_out[0], h_out[1]], dim=1)
        # return output, h_out.squeeze()


class DecoderRNN(nn.Module):
    def __init__(self, voc, embed_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        self.n_layers = n_layers

        self.embed = nn.Embedding(voc.size, embed_size)
        self.gru1 = nn.GRUCell(embed_size, hidden_size)
        self.gru2 = nn.GRUCell(hidden_size, hidden_size)
        self.gru3 = nn.GRUCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, voc.size)

    def forward(self, input, h):
        h_out = torch.zeros(h.size()).to(util.dev)
        output = self.embed(input)
        output = h_out[0] = self.gru1(output, h[0])
        output = h_out[1] = self.gru2(output, h[1])
        output = h_out[2] = self.gru3(output, h[2])
        output = self.linear(output)
        return output, h_out

    def init_h(self, batch_size):
        return torch.zeros(3, batch_size, self.hidden_size).to(util.dev)


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.contiguous().view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.contiguous().view(b_size, -1), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class Attention(nn.Module):
    def __init__(self, hidden_size, method='general'):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        if self.method == 'dot':
            energy = encoder_outputs
        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
        elif self.method == 'concat':
            h = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = self.attn(torch.cat([h, encoder_outputs], 2))
            hidden = self.v.repeat(encoder_outputs.size(0), 1)  # [B*1*H]
        energy = torch.bmm(energy, hidden.unsqueeze(2))  # [B*T]
        return F.softmax(energy.transpose(1, 2), dim=-1)

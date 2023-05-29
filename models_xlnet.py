from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch import Tensor
import collections
import math
import copy

from torch.nn.utils.rnn import PackedSequence

torch.manual_seed(1)
np.random.seed(1)

from modeling_xlnet import XLNetModel, XLNetPreTrainedModel
from modeling_utils import PreTrainedModel, prune_linear_layer, SequenceSummary, PoolerAnswerClass, PoolerEndLogits, \
    PoolerStartLogits


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

class GRU(nn.Module):
    def __init__(self, indim, hidim, outdim):
        super(GRU, self).__init__()
        self.indim = indim
        self.hidim = hidim
        self.outdim = outdim
        self.W_zh, self.W_zx, self.b_z = self.get_three_parameters()
        self.W_rh, self.W_rx, self.b_r = self.get_three_parameters()
        self.W_hh, self.W_hx, self.b_h = self.get_three_parameters()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=32)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.Linear = nn.Linear(hidim, outdim)  # 全连接层做输出
        self.reset()

    def permute_hidden(self, hx: Tensor, permutation):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)
    def forward(self, input, hx=None):
        self.cuda()
        input = input.type(torch.float32)
        #print(input.shape)
        bs = input.shape[1]
        #print("bs", bs)
        if torch.cuda.is_available():
            input = input.cuda()
        Y = []
        #Y.cuda()
        if hx is None:
            hx = torch.zeros([bs, self.hidim], dtype=input.dtype, device=input.device)
        h = hx
        #h.cuda()
        for x in input:
            z = torch.sigmoid(h @ self.W_zh + x @ self.W_zx + self.b_z)
            r = torch.sigmoid(h @ self.W_rh + x @ self.W_rx + self.b_r)
            ht = torch.tanh((h * r) @ self.W_hh + x @ self.W_hx + self.b_h)
            h = (1 - z) * h + z * ht
            y = self.Linear(h)
            #print("yyyy:", y.shape)
            Y.append(y)
        y = torch.stack(Y, dim=0)
        y = self.transformer_encoder(y)
        return y, h

    def get_three_parameters(self):
        indim, hidim, outdim = self.indim, self.hidim, self.outdim
        return nn.Parameter(torch.FloatTensor(hidim, hidim)), \
               nn.Parameter(torch.FloatTensor(indim, hidim)), \
               nn.Parameter(torch.FloatTensor(hidim))

    def reset(self):
        stdv = 1.0 / math.sqrt(self.hidim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)



class clinical_xlnet_lstm_FAST(nn.Sequential):

    def __init__(self):
        super(clinical_xlnet_lstm_FAST, self).__init__()

        self.intermediate_size = 1536
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.01
        self.hidden_size_encoder = 768
        self.n_layers = 2
        self.hidden_size_xlnet = 768
        self.encoder = GRU(self.hidden_size_encoder, self.hidden_size_encoder, self.hidden_size_encoder)
        self.decoder = nn.Sequential(
            nn.Dropout(p=self.hidden_dropout_prob),
            nn.Linear(self.hidden_size_encoder, 24),
            nn.ReLU(True),
            # output layer
            nn.Linear(24, 1)
        )

    def forward(self, xlnet_outputs):
        output = self.encoder(xlnet_outputs.permute(1,0,2))

        output = output[0]
        last_layer = output[-1]
        xlnet_outputs = xlnet_outputs.permute(1,0,2)
        for i in range(32):
            last_layer = torch.max(last_layer, output[i])
            last_layer = torch.max(last_layer, xlnet_outputs[i])
        score = self.decoder(last_layer)
        return score


class clinical_xlnet_seq(XLNetPreTrainedModel):

    def __init__(self, config):
        super(clinical_xlnet_seq, self).__init__(config)

        self.hidden_size_xlnet = 768

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size_xlnet, 32),
            nn.ReLU(True),
            # output layer
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, seg_ids, masks):
        output = self.sequence_summary(self.transformer(input_ids, token_type_ids=seg_ids, attention_mask=masks)[0])

        score = self.decoder(output)

        return score, output

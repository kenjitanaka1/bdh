import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
# Adopted from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
# citation in paper
class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

# Adopted from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
# citation in paper
class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden
# Modified from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
# citation in paper
class MySeq2SeqFastAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        trg_dim,
        src_emb_dim,
        trg_emb_dim,
        hidden_dim,
        batch_size,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
        device=torch.device('cpu')
    ):
        """Create all of the layers."""
        super(MySeq2SeqFastAttention, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.trg_dim = trg_dim
        self.src_hidden_dim = hidden_dim
        self.trg_hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        # inputs and outputs are now linear layers instead
        self.src_embedding = nn.Linear(1, src_emb_dim)
        self.trg_embedding = nn.Linear(self.trg_dim, trg_emb_dim)
        # only tested with bidirectional=true
        self.src_hidden_dim = self.src_hidden_dim // 2 \
            if self.bidirectional else self.src_hidden_dim
        # encoder and decoder are LSTM networks
        self.encoder = nn.LSTM(
            src_emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            trg_emb_dim,
            self.trg_hidden_dim,
            nlayers_trg,
            batch_first=True,
            dropout=self.dropout
        )
        # passthrough layer is still linear
        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )
        # final out is also still linear
        self.decoder2vocab = nn.Linear(2 * self.trg_hidden_dim, trg_dim)
        # initialize last 2 layers to start with 0 bias
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        # initializes the last 2 linear layers to start with 0 bias
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.to(self.device), c0_encoder.to(self.device)

    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""

        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        src_emb = src_emb.view(1, src_emb.shape[0], self.src_emb_dim)
        trg_emb = trg_emb.view(1, trg_emb.shape[0], self.trg_emb_dim)

        self.h0_encoder, self.c0_encoder = self.get_state(src_emb)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )  # bsize x seqlen x dim

        h_t = src_h_t.view(-1, self.src_hidden_dim * self.num_directions)
        c_t = src_c_t.view(-1, self.src_hidden_dim * self.num_directions)

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.view(
                    self.decoder.num_layers,
                    1,
                    self.trg_hidden_dim
                ),
                c_t.view(
                    self.decoder.num_layers,
                    1,
                    self.trg_hidden_dim
                )
            )
        )  # bsize x seqlen x dim

        # Fast Attention dot product

        # bsize x seqlen_src x seq_len_trg
        alpha = torch.bmm(src_h, trg_h.transpose(1, 2))
        # bsize x seq_len_trg x dim
        alpha = torch.bmm(alpha.transpose(1, 2), src_h)
        # bsize x seq_len_trg x (2 * dim)
        trg_h_reshape = torch.cat((trg_h, alpha), 2)

        trg_h_reshape = trg_h_reshape.view(
            trg_h_reshape.size(0) * trg_h_reshape.size(1),
            trg_h_reshape.size(2)
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """
        Turns a 1 hot array of class probabilities 
        to classifications based on the most likely
        """
        return np.argmax(logits.detach().numpy(), axis=2)
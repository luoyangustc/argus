import torch.nn as nn
import torch


class CompositeLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, multi_gpu=False, first=True):
        super(CompositeLSTM, self).__init__()
        print("config of lstm,inshape(feature length):",
              nIn, ",output feature length:", nHidden)
        self.rnn = nn.LSTM(nIn, nHidden, 2, bidirectional=True, dropout=0.4)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.multi_gpu = multi_gpu
        initrange = 0.08
        print("Initializing Bidirectional LSTM...")
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        residual = input

        if self.multi_gpu:
            self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        output = torch.cat((output, residual), dim=2)

        return output


class MLayerLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nLayer, nClass):
        super(MLayerLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, nLayer, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nClass)

    def forward(self, input):
        residual = input
        recurrent, _ = self.rnn(input)

        T, b, h = recurrent.size()

        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output += residual

        return output


def compositelstm(n_class):
    in_dim = 1664
    n_hidden = 256
    multi_gpu = False
    model = nn.Sequential(
        CompositeLSTM(in_dim, n_hidden, 2 * n_hidden, multi_gpu),
        CompositeLSTM(in_dim + 2 * n_hidden, n_hidden,
                      2 * n_hidden, multi_gpu, False),
        nn.Linear(in_dim + 4 * n_hidden, n_class)
    )
    return model


def lstm_2layer(rnn_conf, n_class):
    in_dim = rnn_conf['n_In']
    n_hidden = rnn_conf['n_Hidden']
    n_layer = rnn_conf['n_Layer']
    model = MLayerLSTM(in_dim, n_hidden, n_layer, n_class)
    return model

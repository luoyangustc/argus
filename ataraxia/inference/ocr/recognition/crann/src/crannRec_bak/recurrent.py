import torch.nn as nn

class CompositeLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, multi_gpu=False):
        super(CompositeLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.multi_gpu = multi_gpu
        initrange = 0.08
        print("Initializing Bidirectional LSTM...")
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        if self.multi_gpu:
            self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class MLayerLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nLayer, nClass):
        super(MLayerLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, nLayer, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nClass)

    def forward(self, input):
        recurrent, _ = self.rnn(input)

        T, b, h = recurrent.size()

        t_rec = recurrent.view(T*b, h)
        output = self.embedding(t_rec)
        
        return output


def compositelstm(rnn_conf, n_class):
    in_dim = rnn_conf['n_In']
    n_hidden = rnn_conf['n_Hidden']
    multi_gpu = rnn_conf['multi_gpu']
    model = nn.Sequential(
        CompositeLSTM(in_dim, n_hidden, n_hidden, multi_gpu),
        CompositeLSTM(n_hidden, n_hidden, n_class, multi_gpu)
    )
    return model


def lstm_2layer(rnn_conf, n_class):
    in_dim = rnn_conf['n_In']
    n_hidden = rnn_conf['n_Hidden']
    n_layer = rnn_conf['n_Layer']
    model = MLayerLSTM(in_dim, n_hidden, n_layer, nClass)
    return model


#TODO Implement Seq2Seq model
#class Seq2Seq(nn.Module):
    




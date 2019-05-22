#!/usr/bin/python
# encoding: utf-8

import convnet as ConvNets
import recurrent as SeqNets
import torch.nn as nn
import torch.nn.parallel


class CRNN(nn.Module):
    def __init__(self, n_class, with_spatial_transform=False):
        super(CRNN, self).__init__()
        self.ngpu = 1

        print('Constructing Densenet169')
        self.cnn = ConvNets.__dict__['DenseNet169']()

        print('Constructing compositelstm')
        self.rnn = SeqNets.__dict__["compositelstm"](n_class)
        self._with_spatial_transform = with_spatial_transform
        if self._with_spatial_transform:
            # localization-network
            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernal_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
            # regressor for the 3*2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]), dtype=torch.float)

    def forward(self, input):
        if self._with_spatial_transform:
            input = self._spatial_transform(input)

        c_feat = self.cnn(input)

        b, c, h, w = c_feat.size()

        assert h == 1, "the height of the conv must be 1"

        c_feat = c_feat.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(c_feat)

        return output

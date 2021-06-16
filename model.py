import torch
import numpy as np
import pandas as pd
import os
import h5py
import torch.nn as nn

__all__ = ['get_conv_bn', 'get_base_layer', 'get_head_layer',
          'get_sq_ex', 'get_dep_sep', 'get_inv_res']


def get_base_layer(in_ch=1, out_ch=32, ks=3, stride=2, padding=1):
    return conv(in_ch=in_ch, out_ch=out_ch, ks=ks, 
               stride=stride, padding=padding, activation=True)


def get_head_layer(in_chans=320, out_chans=1280, ks=1, stride=1,
              avg_out_feats=200, lin_out_feats=1):
    return nn.Sequential(
        conv(in_chans, out_chans, ks, stride, activation=True),
        nn.AdaptiveAvgPool1d(output_size=avg_out_feats),
        nn.Linear(in_features=avg_out_feats, out_features=lin_out_feats))


def get_conv_bn(in_ch=1, out_ch=2, ks=2, stride=2, padding=None):
    return nn.Sequential(
        nn.Conv1d(in_channels = in_ch, out_channels = out_ch,
                 kernel_size = ks, stride = stride, 
                  padding=padding, bias=False),
        nn.BatchNorm1d(num_features = out_ch)
    )

def conv(in_ch, out_ch, ks, stride, padding=0, activation=False):
    res = get_conv_bn(in_ch, out_ch, ks, stride, padding)
    if activation:
        res = nn.Sequential(res, nn.SiLU(inplace=True))
    return res


def get_sq_ex(in_ch= (1,1), out_ch= (2,2), ks= (2,2), stride= (2,2)):
    return nn.Sequential(
        nn.Conv1d(in_channels= in_ch[0], out_channels= out_ch[0], 
                  kernel_size= ks[0], stride= stride[0]),
        nn.SiLU(),
        nn.Conv1d(in_channels= in_ch[1], out_channels= out_ch[1], 
                  kernel_size= ks[1], stride= stride[1])
    )

def get_dep_sep(in_ch, out_ch, ks=3, mid_ch=8):
    return nn.Sequential(
        conv(in_ch=in_ch, out_ch=in_ch, ks=ks, stride=1, activation=True),
        get_sq_ex(in_ch=(in_ch, mid_ch), 
                  out_ch=(mid_ch, in_ch)),
        conv(in_ch=in_ch, out_ch=out_ch, ks=1, stride=1),
        nn.Identity()
    )

def get_inv_res(in_ch, mid_ch, out_ch, sq_ch=4, ks=1, stride=1, padding=1):
    return nn.Sequential(
        conv(in_ch=in_ch, out_ch=mid_ch, ks=1, stride=1, activation=True),
        conv(in_ch=mid_ch, out_ch=mid_ch, ks=ks, stride=stride, 
             padding=padding, activation=True),
        get_sq_ex((mid_ch,sq_ch), (sq_ch, mid_ch) ,ks=(1,1), stride=(1,1)),
        conv(in_ch=mid_ch, out_ch=out_ch, ks=1, stride=1)
    )
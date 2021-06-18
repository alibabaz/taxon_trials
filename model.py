import torch
import numpy as np
import pandas as pd
import os
import h5py
import torch.nn as nn

__all__ = ['get_conv_bn', 'get_base_layer', 'get_head_layer',
          'get_sq_ex', 'get_dep_sep', 'get_inv_res', 'conv',
          'get_eff_b0']


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


def get_eff_b0(out_feats):
    return nn.Sequential(
        get_base_layer(),
        get_dep_sep(32, 16),

        #layer 1 has two inverted residuals (IRs)
        get_inv_res(in_ch=16, mid_ch=96, out_ch=24,
                   sq_ch=4, ks=3, stride=2, padding=1),
        get_inv_res(in_ch=24, mid_ch=144, out_ch=24,
                   sq_ch=6, ks=3, stride=1, padding=1),

        #layers 2 has 2 IR's
        get_inv_res(in_ch=24, mid_ch=144, out_ch=40,
                   sq_ch=6, ks=5, stride=2, padding=2),
        get_inv_res(in_ch=40, mid_ch=240, out_ch=40,
                   sq_ch=10, ks=5, stride=2, padding=2),

        #layer 3 has 3 IR's
        get_inv_res(in_ch=40, mid_ch=240, out_ch=80,
                   sq_ch=10, ks=3, stride=2, padding=1),
        get_inv_res(in_ch=80, mid_ch=480, out_ch=80,
                   sq_ch=20, ks=3, stride=1, padding=1),
        get_inv_res(in_ch=80, mid_ch=480, out_ch=80,
                   sq_ch=20, ks=3, stride=1, padding=1),

        #layer 4 has 3 inverted residuals
        get_inv_res(in_ch=80, mid_ch=480, out_ch=112,
                   sq_ch=20, ks=5, stride=1, padding=2),
        get_inv_res(in_ch=112, mid_ch=672, out_ch=112,
                   sq_ch=28, ks=5, stride=1, padding=2),
        get_inv_res(in_ch=112, mid_ch=672, out_ch=112,
                   sq_ch=28, ks=5, stride=1, padding=2),

        #layer 5 has 4 inverted residuals
        get_inv_res(in_ch=112, mid_ch=672, out_ch=192,
                   sq_ch=28, ks=5, stride=2, padding=2),
        get_inv_res(in_ch=192, mid_ch=1152, out_ch=192,
                   sq_ch=48, ks=5, stride=1, padding=2),
        get_inv_res(in_ch=192, mid_ch=1152, out_ch=192,
                   sq_ch=48, ks=5, stride=1, padding=2),
        get_inv_res(in_ch=192, mid_ch=1152, out_ch=192,
                   sq_ch=48, ks=5, stride=1, padding=2),
        #layer 6 has 1 IR
        get_inv_res(in_ch=192, mid_ch=1152, out_ch=320,
                   sq_ch=48, ks=3, stride=1, padding=1),

        get_head_layer(out_chans=out_feats)
    )
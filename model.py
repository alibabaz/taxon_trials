import torch
import numpy as np
import pandas as pd
import os
import h5py
import torch.nn as nn

__all__ = ['get_conv_bn', 'get_base_layer', 'get_head_layer',
          'get_sq_ex', 'get_dep_sep', 'get_inv_res']

def get_conv_bn(in_ch=1, out_ch=2, ks=2, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels = in_ch, out_channels = out_ch,
                 kernel_size = ks, stride = stride),
        nn.BatchNorm1d(num_features = out_ch)
    )


def get_base_layer(in_chans=1, out_chans=32, ks=3, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels= in_chans, out_channels= out_chans, 
                  kernel_size= ks, stride= stride),
        nn.BatchNorm1d(num_features = out_chans),
        nn.SiLU())


def get_head_layer(in_chans=1, out_chans=32, ks=3, stride=2,
              avg_out_feats=10, lin_out_feats=1):
    return nn.Sequential(
        nn.Conv1d(in_channels= in_chans, out_channels= out_chans, 
                  kernel_size= ks, stride= stride),
        nn.BatchNorm1d(num_features = out_chans),
        nn.SiLU(),
        nn.AdaptiveAvgPool1d(output_size=avg_out_feats),
        nn.Linear(in_features=avg_out_feats, out_features=lin_out_feats))


def get_sq_ex(in_ch= (1,1), out_ch= (2,2), ks= (2,2), stride= (2,2)):
    return nn.Sequential(
        nn.Conv1d(in_channels= in_ch[0], out_channels= out_ch[0], 
                  kernel_size= ks[0], stride= stride[0]),
        nn.SiLU(),
        nn.Conv1d(in_channels= in_ch[1], out_channels= out_ch[1], 
                  kernel_size= ks[1], stride= stride[1])
    )

def get_dep_sep(in_ch, out_ch, ks=3, reduction=6):
    return nn.Sequential(
        get_conv_bn(in_ch=in_ch, out_ch=in_ch*2, ks=ks),
        nn.SiLU(),
        get_sq_ex(in_ch=(in_ch*2, reduction), 
                  out_ch=(reduction, in_ch*2)),
        get_conv_bn(in_ch=in_ch*2, out_ch=out_ch),
        nn.Identity()
    )


def get_inv_res(in_ch, out_ch, ks=3, reduction=4):
    return nn.Sequential(
        get_conv_bn(in_ch=in_ch, out_ch=in_ch*4, ks=1),
        nn.SiLU(),
        get_conv_bn(in_ch=in_ch*4, out_ch=in_ch*4, ks=3),
        get_sq_ex(in_ch=(in_ch*4, reduction),
                 out_ch=(reduction, in_ch*4)),
        get_conv_bn(in_ch=in_ch*4, out_ch=out_ch)
    )
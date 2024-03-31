#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:10:13 2024

@author: comm-kwon
"""
import mlx.nn as nn
import mlx.core as mx
import math

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def __call__(self, inputs):
        return inputs * nn.sigmoid(inputs)

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def __call__(self, inputs):
        outputs, gate = mx.split(inputs,2, axis=self.dim)
        return outputs * nn.sigmoid(gate)


def recover_resolution(inputs):
    outputs = list()

    for idx in range(inputs.shape[1] * 2):
        outputs.append(inputs[:, idx // 2, :])
   
    return mx.array(outputs).transpose(1,0,2)



class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int = 2, padding: int = 0,) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        
        self.multi = out_channels//in_channels
        self.out_channels = out_channels
        
        self.conv = {}
        for i in range(in_channels):
            self.conv['l' + str(i)] = nn.Conv2d(1, self.multi, kernel_size, stride, padding)

    def __call__(self, inputs):
        outputs = []
        if len(inputs.shape)==3:
            inputs = inputs[:,:,:,mx.newaxis]
        for i in range(inputs.shape[-1]):
            outputs += [self.conv['l' + str(i)](inputs)]
        
        outputs = mx.array(outputs).squeeze(-1)
        if len(outputs.shape)==5:
            outputs = outputs.transpose(1,2,3,4,0)
            outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],outputs.shape[2],-1)
        elif len(outputs.shape)==4:
            outputs = outputs.transpose(1,2,3,0)
        return outputs
    
class DepthwiseConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = []
        self.padding = padding
        self.depth_multi = out_channels//in_channels
        self.conv = {}
        for i in range(self.depth_multi):
            self.conv['l' + str(i)] =  nn.Conv2d(1, self.depth_multi, (kernel_size,1), (stride,1), bias = bias)

    def __call__(self, inputs):
        outputs = []
        
        if len(inputs.shape)==3:
            inputs = inputs[:,:,:,mx.newaxis]
        
        inputs = mx.pad(inputs, ((0,0), self.padding, (0,0), (0,0)))
        for i in range(self.depth_multi):
            outputs += [self.conv['l' + str(i)](inputs)]
        
        batch_size, lengths = outputs[0].shape[0:2]
        return mx.array(outputs).reshape(batch_size, lengths,-1)
    
    
class PointwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias, )

    def __call__(self, inputs) :
        return self.conv(inputs)

class TimeReductionLayer_2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super(TimeReductionLayer_2d, self).__init__()
        self.sequential = nn.Sequential(
            DepthwiseConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding = (kernel_size - stride,kernel_size - stride),
            ),
            Swish(),
        )

    def __call__(self, inputs):
        outputs = self.sequential(inputs).squeeze(-1)
        batch_size, subsampled_lengths, _ = outputs.shape
        outputs = outputs.reshape(batch_size, subsampled_lengths,-1)
        return outputs
    


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.init.glorot_uniform()(mx.zeros((self.num_heads,1,self.d_head)))
        
        self.rotary = nn.RoPE(self.d_head, traditional = True)
        
        self.out_proj = nn.Linear(d_model, d_model)
        

    def __call__(self, query, key, value, mask = None,):
        batch_size = value.shape[0]

        query = self.query_proj(query).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        key = self.key_proj(key).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        value = self.value_proj(value).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        
        query = self.rotary(query)
        key = self.rotary(key)
        
        content_score = mx.matmul((query + self.u_bias), key.transpose(0,1,3, 2))
        score = content_score / self.sqrt_dim

        if mask is not None:
            score  = score + mask

        attn = mx.softmax(score, -1)
        attn = self.dropout(attn)

        context = mx.matmul(attn, value).transpose(0,2,1,3)
        context = context.reshape(batch_size, -1, self.d_model)

        return self.out_proj(context)

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Tuple, Union
from rotary_embedding_torch import RotaryEmbedding


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple) -> None:
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
    
def recover_resolution(inputs: Tensor) -> Tensor:
    outputs = list()

    for idx in range(inputs.size(1) * 2):
        outputs.append(inputs[:, idx // 2, :])
   
    return torch.stack(outputs, dim=1)


class DepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


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
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
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
                padding = (1,1),
            ),
            Swish(),
        )

    def forward(self, inputs: Tensor):
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

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
        
        self.rotary = RotaryEmbedding(int(self.d_head))

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.rotary.rotate_queries_or_keys(self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).permute(0,2,1,3)).permute(0,2,1,3)
        key = self.rotary.rotate_queries_or_keys(self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3))
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        score = content_score / self.sqrt_dim
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)


import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
from layer import Swish, GLU, DepthwiseConv1d, PointwiseConv1d, MultiHeadAttention, recover_resolution, TimeReductionLayer_2d, Transpose


class Model(nn.Module):
    def __init__(self, d_input, d_output: int = 80,  d_layers = 1, d_model: int = 512, e_layers: int = 4, conv_expansion_factor: int = 2,
         ff_dropout: float = 0.1, nhead: int = 8, conv_dropout: float = 0.1, kernel_size: int = 31,  ff_expansion_factor: int = 4,
           attention_dropout_p: float = 0.1, half_step_residual: bool = False
    ) -> None:
        super(Model, self).__init__()
          
        self.emb_in = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.Dropout(p= ff_dropout),
        )
        self.d_layers = d_layers
        self.encoder = SqueezeformerEncoder(encoder_dim=d_model, num_layers=e_layers,
            num_attention_heads=nhead,
            feed_forward_expansion_factor=ff_expansion_factor, conv_expansion_factor=conv_expansion_factor,
             feed_forward_dropout_p=ff_dropout,
            attention_dropout_p=attention_dropout_p, conv_dropout_p=conv_dropout,
            conv_kernel_size=kernel_size, half_step_residual=half_step_residual
        )
        if not (d_layers == 0):
            self.rnn = nn.LSTM(d_model, d_model, d_layers, batch_first = True)
            self.fc = nn.Linear(d_model, d_output)
        else:
            self.fc = nn.Linear(d_model, d_output)

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return self.encoder.count_parameters()

    def forward(self, inputs: Tensor) :
        inputs = self.emb_in(inputs)       
        output = self.encoder(inputs)
        if not (self.d_layers ==0):
            output, _ = self.rnn(output)

        outputs = self.fc(output)
        return outputs

    
class SqueezeformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_index: int = 7,
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = False,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.reduce_layer_index = int(num_layers/2) -1 + int(num_layers%2)#0 #int(num_layers/2) #int(num_layers/2)-1 #reduce_layer_index num_layers#
        self.recover_layer_index =  num_layers-1 #num_layers-1 #recover_layer_index
        
        self.time_reduction_layer = TimeReductionLayer_2d()
        self.time_reduction_proj = nn.Linear((encoder_dim ) // 2, encoder_dim)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
                
        self.recover_tensor = None
        
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx < reduce_layer_index:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                        half_step_residual=half_step_residual,
                    )
                )
            elif reduce_layer_index <= idx < recover_layer_index:
                self.layers.append(
                    ResidualConnectionModule(
                        module=SqueezeformerBlock(
                            encoder_dim=encoder_dim,
                            num_attention_heads=num_attention_heads,
                            feed_forward_expansion_factor=feed_forward_expansion_factor,
                            conv_expansion_factor=conv_expansion_factor,
                            feed_forward_dropout_p=feed_forward_dropout_p,
                            attention_dropout_p=attention_dropout_p,
                            conv_dropout_p=conv_dropout_p,
                            conv_kernel_size=conv_kernel_size,
                            half_step_residual=half_step_residual,
                        )
                    )
                )
            else:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                        half_step_residual=half_step_residual,
                    )
                )

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return sum([p.numel for p in self.parameters()])

    def forward(self, inputs: Tensor):
        outputs = inputs
        for idx, layer in enumerate(self.layers):
            if idx == self.reduce_layer_index:
                self.recover_tensor = outputs
                outputs= self.time_reduction_layer(outputs)
                outputs = self.time_reduction_proj(outputs)
            
            if idx == self.recover_layer_index:
                outputs = recover_resolution(outputs)
                outputs = self.time_recover_layer(outputs)
                outputs += self.recover_tensor

            outputs = layer(outputs)

        return outputs


class SqueezeformerBlock(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = False,
    ):
        super(SqueezeformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            nn.LayerNorm(encoder_dim),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
            ResidualConnectionModule(
                module=ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            nn.LayerNorm(encoder_dim),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
    


class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + inputs


class FeedForwardModule(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        #assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        #assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"
        out_channels = in_channels# * expansion_factor
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(out_channels),
            Swish(),
            PointwiseConv1d(out_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p)
            )
        
    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)
    
    
class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, rpe = False):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, inputs: Tensor, mask = None):
        outputs = self.attention(inputs, inputs, inputs, mask=mask)
        return self.dropout(outputs)
    

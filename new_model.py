import numpy as np
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchsummary
from math import log as ln
from einops import rearrange


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level=noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding
  
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x
  
class HNFBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        
        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size//4, 3, dilation=dilation, padding=1*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 5, dilation=dilation, padding=2*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 9, dilation=dilation, padding=4*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 15, dilation=dilation, padding=7*dilation, padding_mode='reflect'),
        ])
        
        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        
        self.norm = nn.InstanceNorm1d(hidden_size//2)
        
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        
    def forward(self, x):
        residual = x
        
        filts = []
        for layer in self.filters:
            filts.append(layer(x))
            
        filts = torch.cat(filts, dim=1)
        
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        
        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.2)
        
        filts = F.leaky_relu(self.conv_2(filts), 0.2)
        
        return filts + residual

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 16, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoding = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, padding_mode='reflect')
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, padding_mode='reflect')
        self.attn = PreNorm(hidden_size, Attention(hidden_size))
    
    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        x = self.output_conv(x)
        x = self.attn(x)
        return x
    

class ConditionalModel(nn.Module):
    def __init__(self, feats=64):
        super(ConditionalModel, self).__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        
        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        
        self.embed = PositionalEncoding(feats)
        
        self.bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])
        
        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')
        
    def forward(self, x,  noise_scale, cond):
        noise_embed = self.embed(noise_scale)
        # print(f"noise embed shape: {noise_embed.shape}")
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed)) 
        
        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond)+x
        
        return self.conv_out(cond)

if __name__ == '__main__':
    model = ConditionalModel(feats=64)
    input_size = (1, 10000)
    batch_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtypes = [torch.float32]
    result = summary(model, input_size=input_size)


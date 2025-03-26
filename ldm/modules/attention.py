from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pickle
import os
import torchvision

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.q = None
        self.k = None
        self.v = None

    def forward(self,
                x,
                context=None,
                mask=None,
                q_input_injected=None,
                k_input_injected=None,
                k_ref_injected=None,
                k_refL_injected=None,
                v_ref_injected=None,
                injection_config=None,):
        self.attn = None
        h = self.heads
        b = x.shape[0]
        q_mix = 0.
        if injection_config is not None:
            if x.shape[-1] == 320:
                gamma = injection_config['gamma1']
                beta = injection_config['beta1']
            else:
                gamma = injection_config['gamma2']
                beta = injection_config['beta2']

        if q_input_injected is None:
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

            context = default(context, x)

            k = self.to_k(context)
            k = rearrange(k, 'b m (h d) -> (b h) m d', h=h)

            v = self.to_v(context)
            v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)

            self.q = q
            self.k = k
            self.v = v

            sim = einsum('b i d, b j d -> b i j', q, k)

            sim *= self.scale
            attn = sim.softmax(dim=-1)
            self.attn = attn
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        else:
            ### gray-to-gray attention ###
            q_uncond = q_input_injected
            q = torch.cat([q_uncond]*b)

            k_uncond = k_refL_injected
            k = torch.cat([k_uncond]*b ,dim=0)

            v_uncond = v_ref_injected
            v = torch.cat([v_uncond]*b ,dim=0)

            sim_g2g = einsum('b i d, b j d -> b i j', q, k)
            sim_g2g *= self.scale

            ### colored-to-color attention ###
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

            k_uncond = k_ref_injected
            k = torch.cat([k_uncond]*b ,dim=0)

            sim_c2c = einsum('b i d, b j d -> b i j', q, k)
            sim_c2c *= self.scale

            ### dual attention ###
            if injection_config['pre_softmax']:
                attn_dual = sim_g2g.softmax(dim=-1) * gamma + sim_c2c.softmax(dim=-1) * (1 - gamma)
            else:
                sim = sim_g2g * gamma + sim_c2c * (1 - gamma)
                attn_dual = sim.softmax(dim=-1)
            self.attn = attn_dual
            out = einsum('b i j, b j d -> b i d', attn_dual, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            ### self-attention injection ###
            if beta != 0:
                q = q_input_injected
                k = k_input_injected

                context = default(context, x)
                v = self.to_v(context)
                v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)

                sim_self = einsum('b i d, b j d -> b i j', q, k)

                sim_self *= self.scale
                attn_self = sim_self.softmax(dim=-1)
                out_self = einsum('b i j, b j d -> b i d', attn_self, v)
                out_self = rearrange(out_self, '(b h) n d -> b n (h d)', h=h)

                out = out * (1 - beta) + out_self * beta

            return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self,
                x,
                context=None,
                self_attn_q_input_injected=None,
                self_attn_k_input_injected=None,
                self_attn_k_ref_injected=None,
                self_attn_k_refL_injected=None,
                self_attn_v_ref_injected=None,
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_q_input_injected,
                                          self_attn_k_input_injected,
                                          self_attn_k_ref_injected,
                                          self_attn_k_refL_injected,
                                          self_attn_v_ref_injected,
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_q_input_injected=None,
                 self_attn_k_input_injected=None,
                 self_attn_k_ref_injected=None,
                 self_attn_k_refL_injected=None,
                 self_attn_v_ref_injected=None,
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       q_input_injected=self_attn_q_input_injected,
                       k_input_injected=self_attn_k_input_injected,
                       k_ref_injected=self_attn_k_ref_injected,
                       k_refL_injected=self_attn_k_refL_injected,
                       v_ref_injected=self_attn_v_ref_injected,
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_q_input_injected=None,
                self_attn_k_input_injected=None,
                self_attn_k_ref_injected=None,
                self_attn_k_refL_injected=None,
                self_attn_v_ref_injected=None,
                injection_config=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_q_input_injected=self_attn_q_input_injected,
                      self_attn_k_input_injected=self_attn_k_input_injected,
                      self_attn_k_ref_injected=self_attn_k_ref_injected,
                      self_attn_k_refL_injected=self_attn_k_refL_injected,
                      self_attn_v_ref_injected=self_attn_v_ref_injected,
                      injection_config=injection_config)


        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

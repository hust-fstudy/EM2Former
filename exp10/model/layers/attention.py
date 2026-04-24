# -*- coding: utf-8 -*-
# @Time: 2026/3/25
# @File: attention.py
# @Author: fwb
# -*- coding: utf-8 -*-
# @Time: 2025/7/12
# @File: attention.py
# @Author: fwb
import math
import torch
import torch.nn as nn
from torch_geometric.data import Data
from timm.layers import DropPath
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from model.layers.sequential import DataModule, DataSequential
from model.layers.dilated_conv_block import DilatedConvBlock
from model.components import DataDict
from model.batch_tools import offset2bincount
from libs.pointrope.rope import ROPE

try:
    import flash_attn
except ImportError:
    flash_attn = None


class DSConv1d(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
    ):
        super(DSConv1d, self).__init__()
        self.depth_wise = nn.Conv1d(
            in_channels=in_chs,
            out_channels=in_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_chs,
            bias=bias
        )
        self.point_wise = nn.Conv1d(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.batch_norm = nn.BatchNorm1d(out_chs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class EMamba(DataModule):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.act = nn.SiLU()
        self.ds_conv1d_x = DSConv1d(
            in_chs=self.d_model // 2,
            out_chs=self.d_inner // 2,
            kernel_size=d_conv,
            padding='same',
            bias=conv_bias // 2
        )
        self.ds_conv1d_z = DSConv1d(
            in_chs=self.d_model // 2,
            out_chs=self.d_inner // 2,
            kernel_size=d_conv,
            padding='same',
            bias=conv_bias // 2
        )

        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = DSConv1d(
            in_chs=self.d_inner,
            out_chs=self.d_model,
            kernel_size=d_conv,
            padding='same',
            bias=conv_bias // 2
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape

        xz = rearrange(hidden_states, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())

        x = self.act(self.ds_conv1d_x(x))
        z = self.act(self.ds_conv1d_z(z))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None
        )
        y = torch.cat([y, z], dim=1)
        y = self.out_proj(y)
        out = rearrange(y, "b d l -> b l d")

        return out


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class HybridAttention(DataModule):
    def __init__(
            self,
            chs,
            num_heads,
            patch_size,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            attn_type='M',
            enable_rpe=False,
            enable_flash=True,
            enable_rope=True,
            upcast_attention=True,
            upcast_softmax=True,
            d_state=8,
            d_conv=3,
            expand=1,
    ):
        super().__init__()
        assert chs % num_heads == 0
        self.chs = chs
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = qk_scale or (chs // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.enable_rope = enable_rope
        self.attn_type = attn_type
        if enable_flash:
            assert (
                    enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                    upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                    upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)
        if attn_type == 'T':  # transformer.
            self.qkv = torch.nn.Linear(chs, chs * 3, bias=qkv_bias)
            self.softmax = torch.nn.Softmax(dim=-1)
            self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None
            self.rope = ROPE() if self.enable_rope else None
        elif attn_type == 'M':  # mamba.
            self.patch_size_max = patch_size
            self.attn = EMamba(d_model=chs, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.attn = None
            print(f"Attention {self.attn_type} does not exist!")

        # FFN.
        self.proj = torch.nn.Linear(chs, chs)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                    torch.div(
                        bincount + self.patch_size - 1,
                        self.patch_size,
                        rounding_mode="trunc",
                    )
                    * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                    _offset_pad[i + 1]
                    - self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                                                           - self.patch_size
                        ]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash or self.attn_type == 'M':
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.chs

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = pad.clone()
        inverse = unpad.clone()

        if self.attn_type == 'T':
            # padding and reshape feat and batch for serialized point patch
            qkv = self.qkv(point.x)[order]
            if self.enable_rope:
                pos = point.grid_coord[order]
                pos = pos.reshape(-1, 3).unsqueeze(0)
                q, k, v = qkv.half().chunk(3, dim=-1)
                q = q.reshape(-1, H, C // H).transpose(0, 1)[None]
                k = k.reshape(-1, H, C // H).transpose(0, 1)[None]
                q = self.rope(q.float(), pos).to(q.dtype)
                k = self.rope(k.float(), pos).to(k.dtype)
                qkv_rope = torch.stack([
                    q.squeeze(0).transpose(0, 1),
                    k.squeeze(0).transpose(0, 1),
                    v.reshape(-1, H, C // H)
                ], dim=1)  # [N, 3, H, head_dim]
            else:
                qkv_rope = qkv.clone()

            if not self.enable_flash:
                # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
                q, k, v = (
                    qkv_rope.reshape(-1, C).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
                )
                # attn
                if self.upcast_attention:
                    q = q.float()
                    k = k.float()
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
                if self.enable_rpe:
                    attn = attn + self.rpe(self.get_rel_pos(point, order))
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(qkv.dtype)
                feat = (attn @ v.to(attn.dtype)).transpose(1, 2).reshape(-1, C)
            else:
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv_rope,
                    cu_seqlens,
                    max_seqlen=self.patch_size,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)
                feat = feat.to(qkv.dtype)
        elif self.attn_type == 'M':
            feat = point.x[order]
            feat = self.attn(feat.reshape(-1, K, C)).reshape(-1, C)
        else:
            feat = None
            print(f"Attention {self.attn_type} does not exist!")
        feat = feat[inverse]
        # FFN.
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.x = feat
        return point


class MLP(nn.Module):
    def __init__(
            self,
            in_chs,
            hidden_chs=None,
            out_channels=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_chs
        hidden_chs = hidden_chs or in_chs
        self.fc1 = nn.Linear(in_chs, hidden_chs)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_chs, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(DataModule):
    def __init__(
            self,
            chs,
            down_kernel_size=(3, 3, 3),
            down_stride=(1, 2, 2),
            num_sparse_block=(2, 1, 1),
            xy_only=False,
            num_heads=2,
            patch_size=48,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            pre_norm=True,
            order_index=0,
            attn_type='M',
            enable_rpe=False,
            enable_flash=True,
            enable_rope=True,
            upcast_attention=False,
            upcast_softmax=False,
            d_state=8,
            d_conv=3,
            expand=1
    ):
        super().__init__()
        self.chs = chs
        self.pre_norm = pre_norm

        self.dc_block = DilatedConvBlock(
            dim=chs,
            down_kernel_size=down_kernel_size,
            down_stride=down_stride,
            num_sparse_block=num_sparse_block,
            xy_only=xy_only,
        )

        self.norm1 = DataSequential(norm_layer(chs))
        self.attn = HybridAttention(
            chs=chs,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            attn_type=attn_type,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            enable_rope=enable_rope,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm2 = DataSequential(norm_layer(chs))
        self.mlp = DataSequential(
            MLP(
                in_chs=chs,
                hidden_chs=int(chs * mlp_ratio),
                out_channels=chs,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = DataSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, data_dict: DataDict):
        shortcut = data_dict.x
        data_dict = self.dc_block(Data(**data_dict), data_dict.grid_size)
        data_dict.x = shortcut + data_dict.x
        shortcut = data_dict.x
        if self.pre_norm:
            data_dict = self.norm1(data_dict)
        data_dict = self.drop_path(self.attn(data_dict))
        data_dict.x = shortcut + data_dict.x
        if not self.pre_norm:
            data_dict = self.norm1(data_dict)

        shortcut = data_dict.x
        if self.pre_norm:
            data_dict = self.norm2(data_dict)
        data_dict = self.drop_path(self.mlp(data_dict))
        data_dict.x = shortcut + data_dict.x
        if not self.pre_norm:
            data_dict = self.norm2(data_dict)
        data_dict.sparse_conv_feat = data_dict.sparse_conv_feat.replace_feature(data_dict.x)

        return data_dict


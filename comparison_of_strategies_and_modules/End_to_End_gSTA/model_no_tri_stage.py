import math
from einops import rearrange
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse


class DWConv(nn.Module):
    def __init__(self, dim):  # 4096
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MixMlp(nn.Module):
    def __init__(self,
                 in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):  # 512,4096, GELU,0
        super().__init__()
        out_features = out_features or in_features  # 512
        hidden_features = hidden_features or in_features  # 4096
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)  # CFF: Convlutional feed-forward network  4096
        self.act = act_layer()  # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.drop(x)

        return x  # 1, 512, 20, 45


class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, depth=None, dilation=None):  # 46*64, [21,49]
        super().__init__()
        if dilation is None:
            dilation = (3, 3)
        d_k_1 = 2 * dilation[0] - 1  # 5
        d_p_1 = (d_k_1 - 1) // 2  # 2
        dd_k_1 = kernel_size[0] // dilation[0] + ((kernel_size[0] // dilation[0]) % 2 - 1)  # 7
        dd_p_1 = dilation[0] * (dd_k_1 - 1) // 2  # 9

        d_k_2 = 2 * dilation[1] - 1  # 13
        d_p_2 = (d_k_2 - 1) // 2  # 6
        dd_k_2 = kernel_size[1] // dilation[1] + ((kernel_size[1] // dilation[1]) % 2 - 1)  # 7
        dd_p_2 = dilation[1] * (dd_k_2 - 1) // 2  # 21

        self.mlp = nn.Sequential(nn.GELU(),
                                 nn.Linear(depth, dim)
                                 )
        # self.norm = nn.GroupNorm(1, dim)
        self.conv0 = nn.Conv2d(dim, dim, (d_k_1, d_k_2), padding=(d_p_1, d_p_2), groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, (dd_k_1, dd_k_2), stride=1, padding=(dd_p_1, dd_p_2), groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)  # 46*64, 46*64*2

    def forward(self, x):  # 1,46*64.20,45   1,64
        # depth_emb = depth_emb.permute(1, 0)
        # depth_emb = self.mlp(depth_emb)  # 1,46*64
        attn = self.conv0(x)                   # 1,46*512.20,45
        # attn = self.norm(attn)
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution

        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x  # 1,46*64,20,45


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=None, depth=None, attn_shortcut=True):
        super().__init__()  # 46*64,[21,49]

        if kernel_size is None:
            kernel_size = (21, 21)
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv   46*64,46*64
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = AttentionModule(d_model, depth, kernel_size)  # 46*64, [21,49]
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):  # 1,46*512.20,45   1,512
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x  # 1,46*512,20,45


class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self, dim, kernel_size=None, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, depth=None, act_layer=nn.GELU):  # 46*64,[21,49],4,0,i
        super().__init__()
        if kernel_size is None:
            kernel_size = (21, 21)
        self.norm1 = nn.BatchNorm2d(dim)  # 46*64
        self.attn = SpatialAttention(dim, depth, kernel_size)  # 46*64,[21,49]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 46*64,46*64*4, GELU,0

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):  # 1,46*64.20,45   1,64
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x  # 1,46*64.20,45


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=None,
                 stride=1,
                 padding=None,
                 dilation=1,
                 upsampling=False,
                 act_norm=False):  # 512,512, [3,7],1,(T,F,T,F,T,F),(1,3)
        super(BasicConv2d, self).__init__()
        if kernel_size is None:
            kernel_size = (3, 3)
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),  # 512,512*4=4096, [3,7],1,(T,F,T,F,T,F),(1,3)
                nn.PixelShuffle(2)  # 1,512,40,90
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=None,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True):
        super(ConvSC, self).__init__()

        if kernel_size is None:
            kernel_size = (3, 3)
        stride = 2 if downsampling is True else 1
        padding_1 = (kernel_size[0] - stride + 1) // 2  # 1
        padding_2 = (kernel_size[1] - stride + 1) // 2  # 3

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=(padding_1, padding_2),
                                act_norm=act_norm)  # 512,512, [3,7],1,(T,F,T,F,T,F),(1,3)

    def forward(self, x):
        y = self.conv(x)
        return y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)  # 1,512,160,360
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)  # 1,512,20,45
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class SinusoidaPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidaPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim / 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        differ=emb.max()-emb.min()
        emb=(emb-emb.min())/differ
        return emb


class Depth_MLP(nn.Module):
    def __init__(self, dim):
        super(Depth_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidaPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self, dim, input_resolution=None, layer_i=0, mlp_ratio=4., drop=0., drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(dim, input_resolution, num_heads=8, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[0]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[0], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[0]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path1(x)
        x = x + self.drop_path1(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class Thermohaline_Model(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4,
                 mlp_ratio=4, drop=0.0, drop_path=0.1, spatio_kernel_enc=None,
                 spatio_kernel_dec=None, Depth_out1=None, Depth_out2=None, Depth_out3=None):
        super(Thermohaline_Model, self).__init__()
        if spatio_kernel_enc is None:
            spatio_kernel_enc = (3, 3)

        if spatio_kernel_dec is None:
            spatio_kernel_dec = (3, 3)
        T, C, H, W = in_shape  # T is pre_seq_length  1,14,160,360
        H_, W_ = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))  # downsample 1 / 2**(N_S/2)  20,45   N_S=6



        self.d1=Depth_out1
        self.d2=Depth_out2
        self.d3=Depth_out3

        # C_out=32
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)  # 14, 64, 6, [3,7]

        self.dec1 = Decoder(hid_S, C, N_S, spatio_kernel_dec)  # 64,32, 6, [3,7]
        self.dec2 = Decoder(hid_S, C, N_S, spatio_kernel_dec)
        self.dec3 = Decoder(hid_S, C, N_S, spatio_kernel_dec)

        self.translator1 = nn.Sequential(
            nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            # GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            # GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out1, drop=drop, drop_path=drop_path),
            nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0)
        )

        self.translator2 = nn.Sequential(
            nn.Conv2d((T + Depth_out1) * hid_S, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, hid_T, kernel_size=1, stride=1, padding=0),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out2, drop=drop, drop_path=drop_path),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out2, drop=drop, drop_path=drop_path),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out2, drop=drop, drop_path=drop_path),
            nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0)
        )

        self.translator3 = nn.Sequential(
            nn.Conv2d((T + Depth_out2) * hid_S, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, hid_T, kernel_size=1, stride=1, padding=0),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            # GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            # GASubBlock(dim=hid_T, mlp_ratio=mlp_ratio, depth=Depth_out3, drop=drop, drop_path=drop_path),
            nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0)
        )

        # self.atten1 = nn.Sequential(nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
        #                             SwinSubBlock(dim=hid_T, input_resolution=(H_, W_), layer_i=1, mlp_ratio=mlp_ratio,
        #                                          drop=drop, drop_path=drop_path))
        self.atten2 = nn.Sequential(nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
                                    SwinSubBlock(dim=hid_T, input_resolution=(H_, W_), layer_i=1, mlp_ratio=mlp_ratio,
                                                 drop=drop, drop_path=drop_path),
                                    nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0),)
        self.atten3 = nn.Sequential(nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
                                    SwinSubBlock(dim=hid_T, input_resolution=(H_, W_), layer_i=1, mlp_ratio=mlp_ratio,
                                                 drop=drop, drop_path=drop_path),
                                    nn.Conv2d(hid_T,T * hid_S,  kernel_size=1, stride=1, padding=0))

        self.depth_mlp1 = Depth_MLP(dim=hid_S)  # 64
        self.depth_mlp2 = Depth_MLP(dim=hid_S)
        self.depth_mlp3 = Depth_MLP(dim=hid_S)


        self.lp1 = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.lp2 = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.lp3 = Encoder(C, hid_S, N_S, spatio_kernel_enc)



        self.depth1_code = nn.Sequential(
            nn.Conv2d(T, 32, (1, 1), (1, 1)),
            nn.Conv2d(32, 64, (1, 1), (1, 1)),
            nn.Conv2d(64, self.d3, (1, 1), (1, 1)),
            nn.GroupNorm(1,self.d3),
            nn.SiLU(),
        )
        # self.depth2_code = nn.Conv2d(T, Depth_out2, kernel_size=1, stride=1, padding=0)
        # self.depth3_code = nn.Conv2d(T, Depth_out3, kernel_size=1, stride=1, padding=0)

        self.depth1_code_c= nn.Sequential(
            nn.Conv2d(T, 32, (1, 1), (1, 1)),
            nn.Conv2d(32, 64, (1, 1), (1, 1)),
            nn.Conv2d(64, self.d3, (1, 1), (1, 1)),
            nn.GroupNorm(1,self.d3),
            nn.SiLU(),
        )
        # self.depth2_code_c = nn.Conv2d(Depth_out1, Depth_out2, kernel_size=1, stride=1, padding=0)
        # self.depth3_code_c = nn.Conv2d(Depth_out2, Depth_out3, kernel_size=1, stride=1, padding=0)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape  # 1,10,1,180,360
        x = x_raw.reshape(B * T, C, H, W)  # 10,1,180,360

        d1=self.d1
        d2=self.d2
        d3=self.d3

        # depth_emb1 = self.depth_mlp1(depth_out1)  # d1,64
        embed, skip = self.enc(x)  # 10,64,90,180    10,64,180,360
        _, C_, H_, W_ = embed.shape  # 64,90,180
        embed1 = embed.reshape(B, T * C_, H_, W_)  # 1,10*64,90,180

        # atten1 = self.atten1(embed1)  # 1,128,90,180
        # Y1 = self.translator1(atten1, depth_emb1)  # 1,10*64,90,180
        for idx,layer in enumerate(self.translator1):
            if isinstance(layer,nn.Conv2d) and idx==0:
                Y1=layer(embed1)
            elif isinstance(layer,GASubBlock):
                Y1=layer(Y1)
            else:
                Y1=layer(Y1)    # 1,10*64,90,180



        Y1 = Y1.reshape(B, T, C_, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H_, W_)  # 1*64,10,90,180
        Y1 = self.depth1_code(Y1).reshape(B, C_, d3, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * d3, C_,H_, W_)  # 1*d3,64,90,180

        skip1 = skip.reshape(B, T, C_, H, W).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H, W)  # 1*64,10,180,360
        skip1 = self.depth1_code_c(skip1).reshape(B, C_, d3, H, W).permute(0, 2, 1, 3, 4).reshape(B * d3,
                                                                                                        C_, H, W)  # 1*d3,64,180,360

        Y1 = self.dec1(Y1, skip1) # 1*d1,1,180,360
        Y1 = Y1.reshape(B, d3, C, H, W)  # 1,d1,1,180,360

        # if len(input_tensor)==0:
        #     y_raw2=Y1
        # else:
        #     y_raw2=input_tensor[:,:d1,:,:,:]

        # y_raw2 = Y1
        #
        # y_raw2 = y_raw2.reshape(B * d1, C, H, W)  # 1*d1,1,180,360
        #
        # # depth_emb2 = self.depth_mlp2(depth_out2)  # d2,64
        # embed2, skip2 = self.lp2(y_raw2)  # 1*d1,64,90,180   1*d1,64,180,360
        # embed2 = embed2.reshape(B, d1, C_, H_, W_)  # 1,d1,64,90,180
        # # atten2=self.atten2(embed1)  # 1,10*64,90,180
        # # atten2_c=atten2
        # # atten2=atten2.reshape(B,T,C_,H_,W_)  # 1,10,64,90,180
        # embed_r=embed.reshape(B,T,C_,H_,W_)
        # concat2 = torch.cat([embed_r, embed2], dim=1)  # 1,10+d1,64,90,180
        # concat2 = concat2.reshape(B, (T + d1) * C_, H_, W_)  # 1,(10+d1)*64,90,180
        # # Y2=self.translator2(concat2,depth_emb2)  # 1,10*64,90,180
        #
        # for idx, layer in enumerate(self.translator2):
        #     if isinstance(layer, nn.Conv2d) and (idx==0 or idx ==1):
        #         concat2 = layer(concat2)
        #     elif isinstance(layer, GASubBlock):
        #         Y2 = layer(concat2)
        #     else:
        #         Y2 = layer(Y2)  # 1,10*64,90,180
        # Y2 = Y2.reshape(B, T, C_, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H_, W_)  # 1*64,10,90,180
        # Y2 = self.depth2_code(Y2).reshape(B, C_, d2, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * d2, C_,
        #                                                                                             H_,
        #                                                                                             W_)  # 1*d2,64,90,180
        #
        # skip2 = skip2.reshape(B, d1, C_, H, W).permute(0, 2, 1, 3, 4).reshape(B * C_, d1, H, W)  # 1*64,10,180,360
        # skip2 = self.depth2_code_c(skip2).reshape(B, C_, d2, H, W).permute(0, 2, 1, 3, 4).reshape(B * d2,
        #                                                                                         C_, H,W)  # 1*d2,64,180,360
        #
        # Y2 = self.dec2(Y2, skip2)  # 1*d2,1,180,360
        # Y2 = Y2.reshape(B, d2, C, H, W)  # 1,d2,1,180,360
        #
        # # if len(input_tensor)==0:
        # #     y_raw3=Y2
        # # else:
        # #     y_raw3=input_tensor[:,:d2,:,:,:]
        #
        # y_raw3 = Y2
        #
        # y_raw3 = y_raw3.reshape(B * d2, C, H, W)  # 1*d2,1,180,360
        #
        # # depth_emb3 = self.depth_mlp3(depth_out3)  # d3,64
        # embed3, skip3 = self.lp3(y_raw3)  # 1*d2,64,90,180  1*d2,64,180,360
        # embed3 = embed3.reshape(B, d2, C_, H_, W_)  # 1,d2,64,90,180
        # # atten3 = self.atten3(atten2_c)  # 1,10*64,90,180
        # # atten3 = atten3.reshape(B, T, C_, H_, W_)  # 1,10,64,90,180
        # concat3 = torch.cat([embed_r, embed3], dim=1)  # 1,10+d2,64,90,180
        # concat3 = concat3.reshape(B, (T + d2) * C_, H_, W_)  # 1,(10+d2)*64,90,180
        # # Y3 = self.translator3(concat3, depth_emb3)  # 1,10*64,90,180
        # for idx, layer in enumerate(self.translator3):
        #     if isinstance(layer, nn.Conv2d) and (idx==0 or idx ==1):
        #         concat3 = layer(concat3)
        #     elif isinstance(layer, GASubBlock):
        #         Y3 = layer(concat3)
        #     else:
        #         Y3 = layer(Y3)  # 1,10*64,90,180
        #
        #
        # Y3 = Y3.reshape(B, T, C_, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H_, W_)  # 1*64,10,90,180
        # Y3 = self.depth3_code(Y3).reshape(B, C_, d3, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * d3, C_,
        #                                                                                     H_,
        #                                                                                     W_)  # 1*d3,64,90,180
        #
        # skip3 = skip3.reshape(B, d2, C_, H, W).permute(0, 2, 1, 3, 4).reshape(B * C_, d2, H, W)  # 1*64,10,180,360
        # skip3 = self.depth3_code_c(skip3).reshape(B, C_, d3, H, W).permute(0, 2, 1, 3, 4).reshape(B * d3,
        #                                                                                         C_, H,
        #                                                                                         W)  # 1*d3,64,180,360
        #
        # Y3 = self.dec3(Y3, skip3)  # 1*d2,1,180,360
        # Y3 = Y3.reshape(B, d3, C, H, W)  # 1,d3,1,180,360

        return Y1

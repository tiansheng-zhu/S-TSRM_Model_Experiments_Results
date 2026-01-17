import math
from einops import rearrange
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse






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

class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y
class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y
class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y



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
            MidIncepNet(T * hid_S, hid_T,2),
            nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0)
        )

        self.translator2 = nn.Sequential(
            MidIncepNet((T + Depth_out1) * hid_S, hid_T,2),
            nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0)
        )

        self.translator3 = nn.Sequential(
            MidIncepNet((T + Depth_out2) * hid_S, hid_T,2),
            nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0)
        )

        # self.atten1 = nn.Sequential(nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
        #                             SwinSubBlock(dim=hid_T, input_resolution=(H_, W_), layer_i=1, mlp_ratio=mlp_ratio,
        #                                          drop=drop, drop_path=drop_path))
        # self.atten2 = nn.Sequential(nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
        #                             SwinSubBlock(dim=hid_T, input_resolution=(H_, W_), layer_i=1, mlp_ratio=mlp_ratio,
        #                                          drop=drop, drop_path=drop_path),
        #                             nn.Conv2d(hid_T, T * hid_S, kernel_size=1, stride=1, padding=0),)
        # self.atten3 = nn.Sequential(nn.Conv2d(T * hid_S, hid_T, kernel_size=1, stride=1, padding=0),
        #                             SwinSubBlock(dim=hid_T, input_resolution=(H_, W_), layer_i=1, mlp_ratio=mlp_ratio,
        #                                          drop=drop, drop_path=drop_path),
        #                             nn.Conv2d(hid_T,T * hid_S,  kernel_size=1, stride=1, padding=0))

        self.depth_mlp1 = Depth_MLP(dim=hid_S)  # 64
        self.depth_mlp2 = Depth_MLP(dim=hid_S)
        self.depth_mlp3 = Depth_MLP(dim=hid_S)


        self.lp1 = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.lp2 = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.lp3 = Encoder(C, hid_S, N_S, spatio_kernel_enc)



        self.depth1_code = nn.Conv2d(T, Depth_out1, kernel_size=1, stride=1, padding=0)
        self.depth2_code = nn.Conv2d(T, Depth_out2, kernel_size=1, stride=1, padding=0)
        self.depth3_code = nn.Conv2d(T, Depth_out3, kernel_size=1, stride=1, padding=0)

        self.depth1_code_c= nn.Conv2d(T, Depth_out1, kernel_size=1, stride=1, padding=0)
        self.depth2_code_c = nn.Conv2d(Depth_out1, Depth_out2, kernel_size=1, stride=1, padding=0)
        self.depth3_code_c = nn.Conv2d(Depth_out2, Depth_out3, kernel_size=1, stride=1, padding=0)

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
            elif isinstance(layer,gInception_ST):
                Y1=layer(Y1)
            else:
                Y1=layer(Y1)    # 1,10*64,90,180

        Y1 = Y1.reshape(B, T, C_, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H_, W_)  # 1*64,10,90,180
        Y1 = self.depth1_code(Y1).reshape(B, C_, d1, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * d1, C_,H_, W_)  # 1*d1,64,90,180

        skip1 = skip.reshape(B, T, C_, H, W).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H, W)  # 1*64,10,180,360
        skip1 = self.depth1_code_c(skip1).reshape(B, C_, d1, H, W).permute(0, 2, 1, 3, 4).reshape(B * d1,
                                                                                                        C_, H, W)  # 1*d1,64,180,360

        Y1 = self.dec1(Y1, skip1) # 1*d1,1,180,360
        Y1 = Y1.reshape(B, d1, C, H, W)  # 1,d1,1,180,360

        # if len(input_tensor)==0:
        #     y_raw2=Y1
        # else:
        #     y_raw2=input_tensor[:,:d1,:,:,:]

        y_raw2 = Y1

        y_raw2 = y_raw2.reshape(B * d1, C, H, W)  # 1*d1,1,180,360

        # depth_emb2 = self.depth_mlp2(depth_out2)  # d2,64
        embed2, skip2 = self.lp2(y_raw2)  # 1*d1,64,90,180   1*d1,64,180,360
        embed2 = embed2.reshape(B, d1, C_, H_, W_)  # 1,d1,64,90,180
        # atten2=self.atten2(embed1)  # 1,10*64,90,180
        # atten2_c=atten2
        # atten2=atten2.reshape(B,T,C_,H_,W_)  # 1,10,64,90,180
        embed_r=embed.reshape(B,T,C_,H_,W_)
        concat2 = torch.cat([embed_r, embed2], dim=1)  # 1,10+d1,64,90,180
        concat2 = concat2.reshape(B, (T + d1) * C_, H_, W_)  # 1,(10+d1)*64,90,180
        # Y2=self.translator2(concat2,depth_emb2)  # 1,10*64,90,180

        for idx, layer in enumerate(self.translator2):
            if isinstance(layer, nn.Conv2d) and (idx==0 or idx ==1):
                concat2 = layer(concat2)
            elif isinstance(layer, gInception_ST):
                Y2 = layer(concat2)
            else:
                Y2 = layer(Y2)  # 1,10*64,90,180
        Y2 = Y2.reshape(B, T, C_, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H_, W_)  # 1*64,10,90,180
        Y2 = self.depth2_code(Y2).reshape(B, C_, d2, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * d2, C_,
                                                                                                    H_,
                                                                                                    W_)  # 1*d2,64,90,180

        skip2 = skip2.reshape(B, d1, C_, H, W).permute(0, 2, 1, 3, 4).reshape(B * C_, d1, H, W)  # 1*64,10,180,360
        skip2 = self.depth2_code_c(skip2).reshape(B, C_, d2, H, W).permute(0, 2, 1, 3, 4).reshape(B * d2,
                                                                                                C_, H,W)  # 1*d2,64,180,360

        Y2 = self.dec2(Y2, skip2)  # 1*d2,1,180,360
        Y2 = Y2.reshape(B, d2, C, H, W)  # 1,d2,1,180,360

        # if len(input_tensor)==0:
        #     y_raw3=Y2
        # else:
        #     y_raw3=input_tensor[:,:d2,:,:,:]

        y_raw3 = Y2

        y_raw3 = y_raw3.reshape(B * d2, C, H, W)  # 1*d2,1,180,360

        # depth_emb3 = self.depth_mlp3(depth_out3)  # d3,64
        embed3, skip3 = self.lp3(y_raw3)  # 1*d2,64,90,180  1*d2,64,180,360
        embed3 = embed3.reshape(B, d2, C_, H_, W_)  # 1,d2,64,90,180
        # atten3 = self.atten3(atten2_c)  # 1,10*64,90,180
        # atten3 = atten3.reshape(B, T, C_, H_, W_)  # 1,10,64,90,180
        concat3 = torch.cat([embed_r, embed3], dim=1)  # 1,10+d2,64,90,180
        concat3 = concat3.reshape(B, (T + d2) * C_, H_, W_)  # 1,(10+d2)*64,90,180
        # Y3 = self.translator3(concat3, depth_emb3)  # 1,10*64,90,180
        for idx, layer in enumerate(self.translator3):
            if isinstance(layer, nn.Conv2d) and (idx==0 or idx ==1):
                concat3 = layer(concat3)
            elif isinstance(layer, gInception_ST):
                Y3 = layer(concat3)
            else:
                Y3 = layer(Y3)  # 1,10*64,90,180


        Y3 = Y3.reshape(B, T, C_, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * C_, T, H_, W_)  # 1*64,10,90,180
        Y3 = self.depth3_code(Y3).reshape(B, C_, d3, H_, W_).permute(0, 2, 1, 3, 4).reshape(B * d3, C_,
                                                                                            H_,
                                                                                            W_)  # 1*d3,64,90,180

        skip3 = skip3.reshape(B, d2, C_, H, W).permute(0, 2, 1, 3, 4).reshape(B * C_, d2, H, W)  # 1*64,10,180,360
        skip3 = self.depth3_code_c(skip3).reshape(B, C_, d3, H, W).permute(0, 2, 1, 3, 4).reshape(B * d3,
                                                                                                C_, H,
                                                                                                W)  # 1*d3,64,180,360

        Y3 = self.dec3(Y3, skip3)  # 1*d2,1,180,360
        Y3 = Y3.reshape(B, d3, C, H, W)  # 1,d3,1,180,360

        return Y1,Y2,Y3

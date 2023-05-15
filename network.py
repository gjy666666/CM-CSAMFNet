import math

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # softplus(x)：ln(mri+exp**x)
        # mri、无上限，但是有下限；2、光滑；M-SPE、非单调
        return x *(torch.tanh(F.softplus(x)))

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvLayer1(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, 3, 2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.lu = Mish()
    def forward(self, x, downsample=None):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        normal = self.lu(out)
        return normal


class ConvLayer2(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride):
        super(ConvLayer2, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.lu = Mish()
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        normal = self.lu(out)
        return normal

class EncodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(EncodeBlock, self).__init__()
        out_channels_def = int(in_channels / 2)
        # reflection_padding = int(np.floor(kernel_size / 2))
        # self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        self.conv1 = GhostModule(in_channels,out_channels_def, dw_size=3, stride=stride)
        self.conv2 = GhostModule(out_channels_def, out_channels, dw_size=3, stride=stride)

    def forward(self, x):
        normal = self.conv1(x)
        normal = self.conv2(normal)
        return normal

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x



class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(inp, init_channels, kernel_size, stride, 0, bias=False),
            # test(),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.ReflectionPad2d(dw_size // 2),
            nn.Conv2d(init_channels, new_channels, dw_size, 1, 0, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            # test(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self,channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class DCBAM(nn.Module):
    def __init__(self,channel):
        super(DCBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        out=x+out
        return out


class CMCSA(nn.Module):
    def __init__(self,channel):
        super(CMCSA, self).__init__()
        self.CA1=ChannelAttentionModule(channel)
        self.CA2=ChannelAttentionModule(channel)
        self.SA1 =SpatialAttentionModule()
        self.SA2 =SpatialAttentionModule()

    def forward(self,I1,I2):
        P1=self.CA1(I1)*I1
        P2 = self.CA2(I2)*I2
        # I1
        F1=self.SA1(P1)*P1
        F2=self.SA1(P2)*P2
        F=F1+F2+I1

        T1=self.SA2(P1)*P1
        T2 = self.SA2(P2) * P2
        T=T1+T2+I2
        return F,T

class CMDAF(nn.Module):
    def __init__(self):
        super(CMDAF, self).__init__()
        self.sigmoid =nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self,feature_vis,feature_ir):
        batch_size, channels, _, _ = feature_vis.size()
        #
        sub_vi_ir=feature_vis-feature_ir
        vis_ir_div=sub_vi_ir*self.sigmoid(self.gap(sub_vi_ir))

        sub_ir_vis=feature_ir-feature_vis
        ir_vis_div=sub_ir_vis*self.sigmoid(self.gap(sub_ir_vis))

        feature_vis+= ir_vis_div
        feature_ir+=  vis_ir_div
        return feature_vis,feature_ir


# def CMDAF(feature_vis,feature_ir):
#     sigmoid=nn.Sigmoid()
#     gap=nn.AdaptiveAvgPool2d(1)
#     batch_size,channels,_,_=feature_vis.size()
#
#     sub_vi_ir=feature_vis-feature_ir
#     vis_ir_div=sub_vi_ir*sigmoid(gap(sub_vi_ir))
#
#     sub_ir_vis=feature_ir-feature_vis
#     ir_vis_div=sub_ir_vis*sigmoid(gap(sub_ir_vis))
#
#     feature_vis+= ir_vis_div
#     feature_ir+=  vis_ir_div
#     return feature_vis,feature_ir

class Fusion_network(nn.Module):
    def __init__(self, channel):
        super(Fusion_network, self).__init__()
        # self.CMCSA = CMCSA(channel)
        self.MRID1_conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        # self.MRID1_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.CBAM_MRI = CBAM(channel)
        self.MRID1_conv2 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)

        self.YD1_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.CBAM_Y = CBAM(channel)
        self.YD1_conv2 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)

        self.YD_conv1 = GhostModule(inp=2 * channel, oup=channel, dw_size=1, stride=1)

        self.MRID2_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.YD2_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.D2_conv1 = GhostModule(inp=2 * channel, oup= channel, dw_size=1, stride=1)
        self.D2_conv3_1 = GhostModule(inp=2 * channel, oup= channel, dw_size=3, stride=1)
        self.D2_conv3_2 = GhostModule(inp=2 * channel, oup= channel, dw_size=3, stride=1)

        # self.MRID1_conv1=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=1,stride=1,padding=0)
        # self.CBAM_MRI=AE(channel)
        # self.MRID1_conv2=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=1,stride=1,padding=0)
        #
        # self.YD1_conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        # self.CBAM_Y = AE(channel)
        # self.YD1_conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        #
        # self.YD_conv1= nn.Conv2d(in_channels=2*channel, out_channels=2*channel, kernel_size=1, stride=1, padding=0)
        #
        # self.MRID2_conv1=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=1,stride=1,padding=0)
        # self.YD2_conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0)
        # self.D2_conv1 = nn.Conv2d(in_channels=2*channel, out_channels=2*channel, kernel_size=1, stride=1, padding=0)
        # self.D2_conv3_1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=3, stride=1, padding=1)
        # self.D2_conv3_2 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=3, stride=1,padding=1)

    def forward(self, MRI, Y):
        # T1, Y = self.CMCSA(T1, Y)

        mri1 = self.MRID1_conv1(MRI)
        mri1 = self.CBAM_MRI(mri1)
        mri1 = self.MRID1_conv2(mri1)
        mri = MRI + mri1

        Y1 = self.YD1_conv1(Y)
        Y1 = self.CBAM_Y(Y1)
        Y1 = self.YD1_conv2(Y1)
        y = Y + Y1
        y1 = torch.cat([mri, y], dim=1)
        T1 = self.YD_conv1(y1)

        P1 = self.MRID2_conv1(MRI)
        P2 = self.YD2_conv1(Y)
        P = torch.cat([P1, P2], dim=1)
        P = self.D2_conv1(P)
        # P = self.D2_conv3_1(P)
        # P = self.D2_conv3_2(P)
        F = T1 + P
        return F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.MRI_conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1,stride=1,padding=0)
        self.Y_conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1,stride=1,padding=0)

        self.MRI_conv1 = EncodeBlock(in_channels=16, out_channels=112, kernel_size=3, stride=1)
        self.Y_conv1 = EncodeBlock(in_channels=16, out_channels=112, kernel_size=3, stride=1)
        self.CMCSA1 = CMDAF()
        self.CMCSA2 = CMDAF()
        self.CMCSA3 = CMDAF()
        self.CMCSA4 = CMDAF()
        #
        # self.CMCSA1 = DCBAM(channel=112)
        # self.CMCSA2 = DCBAM(channel=160)
        # self.CMCSA3 = DCBAM(channel=208)
        # self.CMCSA4 = DCBAM(channel=256)

        # self.CMCSA1 = CMCSA(channel=112)
        # self.CMCSA2 = CMCSA(channel=160)
        # self.CMCSA3 = CMCSA(channel=208)
        # self.CMCSA4 = CMCSA(channel=256)
        #
        self.fusion1 = Fusion_network(channel=112)
        self.fusion2 = Fusion_network(channel=160)
        self.fusion3 = Fusion_network(channel=208)
        self.fusion4 = Fusion_network(channel=256)

        self.MRI_conv2 = EncodeBlock(in_channels=112, out_channels=160, kernel_size=3,stride=1)
        self.Y_conv2 =EncodeBlock(in_channels=112, out_channels=160, kernel_size=3,stride=1)


        self.MRI_conv3 = EncodeBlock(in_channels=160, out_channels=208, kernel_size=3,stride=1)
        self.Y_conv3 = EncodeBlock(in_channels=160, out_channels=208, kernel_size=3,stride=1)


        self.MRI_conv4 =EncodeBlock(in_channels=208, out_channels=256, kernel_size=3,stride=1)
        self.Y_conv4 = EncodeBlock(in_channels=208, out_channels=256, kernel_size=3,stride=1)


    def forward(self,MRI,Y):
        activate=Mish()

        MRI_conv0= activate(self.MRI_conv0(MRI))
        Y_conv0=activate(self.Y_conv0(Y))

        MRI_conv1 = activate(self.MRI_conv1(MRI_conv0))
        Y_conv1= activate(self.Y_conv1(Y_conv0))


        MRI_conv10, Y_conv10=self.CMCSA1(MRI_conv1,Y_conv1)

        MRI_conv11=MRI_conv10+MRI_conv1
        Y_conv11=Y_conv10+Y_conv1
        MRI_conv11=self.pool(MRI_conv11)
        Y_conv11 = self.pool( Y_conv11)
        f1=torch.cat([MRI_conv10, Y_conv10],1)
        # f1= self.fusion1(MRI_conv10, Y_conv10)

        MRI_conv2 = activate(self.MRI_conv2(MRI_conv11))
        Y_conv2 = activate(self.Y_conv2(Y_conv11))
        MRI_conv20, Y_conv20 = self.CMCSA2(MRI_conv2, Y_conv2)
        MRI_conv22 = MRI_conv20 + MRI_conv2
        Y_conv22 = Y_conv20 + Y_conv2
        MRI_conv22= self.pool( MRI_conv22)
        Y_conv22 = self.pool(Y_conv22)
        # f2=self.fusion2(MRI_conv20,Y_conv20)
        f2 = torch.cat([MRI_conv20,Y_conv20], 1)

        MRI_conv3 = activate(self.MRI_conv3(MRI_conv22))
        Y_conv3 = activate(self.Y_conv3(Y_conv22))
        MRI_conv30, Y_conv30 = self.CMCSA3(MRI_conv3, Y_conv3)
        MRI_conv33 = MRI_conv30 + MRI_conv3
        Y_conv33 = Y_conv30 + Y_conv3
        MRI_conv33 = self.pool(MRI_conv33)
        Y_conv33 = self.pool(Y_conv33)
        f3 = torch.cat([MRI_conv30,Y_conv30], 1)
        # f3 = self.fusion3(MRI_conv30, Y_conv30)


        MRI_conv4 = activate(self.MRI_conv4(MRI_conv33))
        Y_conv4 = activate(self.Y_conv4(Y_conv33))
        MRI_conv40, Y_conv40 = self.CMCSA4(MRI_conv4, Y_conv4)
        f4 = self.fusion4(MRI_conv40, Y_conv40)
        f4 = torch.cat([MRI_conv40, Y_conv40], 1)
        return [f1,f2,f3,f4]


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DecodeBlock, self).__init__()
        out_channels_def = int(in_channels / 2)
        self.conv1 = GhostModule(in_channels,out_channels_def, dw_size=1, stride=stride)
        self.conv2 = GhostModule(out_channels_def, out_channels, dw_size=3, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()
        self.DCB30 = DecodeBlock(in_channels=928, out_channels=128,kernel_size=3, stride=1)

        self.DCB20 = DecodeBlock(in_channels=736, out_channels=64,kernel_size=3, stride=1)
        self.DCB21 = DecodeBlock(in_channels=512, out_channels=64,kernel_size=3, stride=1)

        self.DCB10 = DecodeBlock(in_channels=544, out_channels=16,kernel_size=3, stride=1)
        self.DCB11 = DecodeBlock(in_channels=304, out_channels=16,kernel_size=3, stride=1)
        self.DCB12 = DecodeBlock(in_channels=320, out_channels=16,kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.conv1=GhostModule(in_c,out_c, dw_size=3, stride=1)

    def forward(self,F):
        DCB30=self.DCB30(torch.cat([F[2], self.up(F[3])],dim=1))

        DCB20=self.DCB20(torch.cat([F[1],self.up(F[2])],dim=1))
        DCB21 = self.DCB21(torch.cat([F[1],self.up(DCB30),DCB20], dim=1))

        DCB10=self.DCB10(torch.cat([F[0],self.up(F[1])],dim=1))
        DCB11 = self.DCB11(torch.cat([F[0],DCB10,self.up(DCB20)], dim=1))
        DCB12 = self.DCB12(torch.cat([F[0], DCB11, DCB10,self.up(DCB21)], dim=1))

        output=self.conv1(DCB12)
        return output


class DCBAMFusion(nn.Module):
    def __init__(self):
        super(DCBAMFusion, self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self, y_vi_image, ir_image):
        F = self.encoder(y_vi_image, ir_image)
        fused_image = self.decoder(F)
        return fused_image
channel=32

input1 = torch.randn(1, 1, 64, 64)
input2 = torch.randn(1,1, 64, 64)

model=DCBAMFusion()
# # para = sum([np.prod(list(p.size())) for p in model.parameters()])
# # type_size = 4
# # print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
output=model(input1,input2)
print(output.shape)
# params: 1.495756M   8.740108M



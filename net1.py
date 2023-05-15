import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import mean

EPSILON = 1e-10


def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class MultConst(nn.Module):
    def forward(self, input):
        return 255*input


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
            if lef_right%2 == 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 == 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
# class ConvLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(ConvLayer, self).__init__()
#         reflection_padding = int(np.floor(kernel_size / 2))
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#         self.dropout = nn.Dropout2d(p=0.5)
#         # self.is_last = is_last
#
#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         # if self.is_last is False:
#         #     # out = F.normalize(out)
#         out = F.relu(out, inplace=True)
#             # out = self.dropout(out)
#         return out

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
        F = F1 + F2
        # F=F1+F2+I1

        T1=self.SA2(P1)*P1
        T2 = self.SA2(P2) * P2
        # T=T1+T2+I2
        T = T1 + T2
        return F,T
# Dense convolution unit
# class DenseConv2d(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(DenseConv2d, self).__init__()
#         self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)
#
#     def forward(self, x):
#         out = self.dense_conv(x)
#         out = torch.cat([x, out], 1)
#         return out


# Dense Block unit
# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        # denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
        #                ConvLayer(out_channels_def, out_channels, 1, stride)]
        denseblock += [GhostModule(in_channels, out_channels_def, dw_size=kernel_size, stride=stride),
                       GhostModule(out_channels_def, out_channels, dw_size=1, stride=stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels,index):
        #
        super(FusionBlock_res, self).__init__()
        self.fusion=Fusion1_network(channels)
        # self.fusion = FusionBlock_res2(channels)
    def forward(self, x_ir, x_vi):
        out=self.fusion(x_ir, x_vi)
        return out



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


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(inp, init_channels, kernel_size, stride, 0, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.ReflectionPad2d(dw_size // 2),
            nn.Conv2d(init_channels, new_channels, dw_size, 1, 0, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class CBAM(nn.Module):
    def __init__(self,channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        # self.conv=nn.Conv2d(3*channel,channel,kernel_size=3,padding=1)
        # self.conv=GhostModule1(inp=3 * channel, oup=channel, dw_size=3, stride=1)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out0 = out + x
        # out = self.spatial_attention(out) * out
        # out0=out+x
        out1 = self.spatial_attention(x) * x
        out1 = out1 + x
        out=out0+out1+x
        return out

class CB(nn.Module):
    def __init__(self,in_channel,ratio=4):
        super(CB,self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.relu = nn.LeakyReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs

def contrast(x):
    mean_x=mean(x)
    c=torch.sqrt(mean((x-mean_x)**2))
    vector=mean(c)
    F=torch.multiply(vector,x)
    return F


class CBE(nn.Module):
    def __init__(self,channel,ratio=4):
        super(CBE,self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=channel, out_features=channel // ratio, bias=False)
        # relu激活
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU(inplace=True)
        # self.conv=nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0)
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=channel // ratio, out_features=channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        b, c, h, w = inputs.shape
        inputs0=contrast(inputs)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs0)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs

class CEM(nn.Module):
    def __init__(self,channel):
        super(CEM, self).__init__()
        # GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.conv1= GhostModule(inp=channel,oup=channel//4,dw_size=1,stride=1)
        self.conv2 =GhostModule(inp=channel,oup=channel//4,dw_size=3,stride=1)
        self.conv3 =GhostModule(inp=channel,oup=channel//4,dw_size=5,stride=1)
        self.conv4 =GhostModule(inp=channel,oup=channel//4,dw_size=7,stride=1)
        # self.DB = CBE(channel)
        self.DB=CB(channel)

    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x4=self.conv4(x)
        x=torch.cat([x1,x2,x3,x4],dim=1)
        # x = torch.cat([x1, x2, x3], dim=1)
        F=self.DB(x)
        return F

class Fusion1_network(nn.Module):
    def __init__(self, channel):
        super(Fusion1_network, self).__init__()
        self.CMCSA = CMCSA(channel)
        self.CMCSA1 = CMCSA(channel)
        self.CMCSA2 = CMCSA(channel)
        self.CMCSA3 = CMCSA(channel)

        # self.EM=CEM(channel)

        self.MRID1_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.CBAM_MRI = CBAM(channel)

        self.MRID1_conv2 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)

        self.YD1_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.CBAM_Y = CBAM(channel)

        self.YD1_conv2 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.YD_conv1 = GhostModule(inp=2 * channel, oup= 2*channel, dw_size=1, stride=1)

        self.MRID2_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
        self.YD2_conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)

        self.D2_conv1 = GhostModule(inp=2 * channel, oup= 2*channel, dw_size=1, stride=1)
        self.D2_conv3_1 = GhostModule(inp=2 * channel, oup= 2*channel, dw_size=3, stride=1)
        self.D2_conv3_2 = GhostModule(inp=2 * channel, oup= 2*channel, dw_size=3, stride=1)

        self.conv_single=GhostModule(inp=2 * channel, oup= channel, dw_size=3, stride=1)

    def forward(self, MRI, Y):
        # MRI=self.EM(MRI)
        # Y = self.EM(Y)
        # MRI, Y = self.CMCSA(MRI, Y)
        # MRI, Y = self.CMCSA1(MRI, Y)
        # MRI, Y = self.CMCSA2(MRI, Y)

        MRI0, Y0 = self.CMCSA3(MRI, Y)
        MRI=MRI+MRI0
        Y=Y+Y0

        # frist RCBAM
        mri1 = (self.MRID1_conv1(MRI))
        mri1 = self.CBAM_MRI(mri1)
        mri1 = (self.MRID1_conv2(mri1))
        mri = MRI + mri1
        Y1 = (self.YD1_conv1(Y))
        Y1 = self.CBAM_Y(Y1)
        Y1 = (self.YD1_conv2(Y1))
        y = Y + Y1
        y1 = torch.cat([mri, y], dim=1)
        T1 = (self.YD_conv1(y1))

        #second SB
        P1 =(self.MRID2_conv1(MRI))
        P2 = (self.YD2_conv1(Y))
        P0 = torch.cat([P1, P2], dim=1)
        P = (self.D2_conv1(P0))
        P = (self.D2_conv3_1(P))
        P = (self.D2_conv3_2(P))
        #
        F1 = P+T1
        F1=(self.conv_single(F1))
        return F1
# Fusion network, 4 groups of features
class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type

        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp


class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp


class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp


class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# fuison strategy based on nuclear-norm (channel attention form NestFuse)
class Fusion_Nuclear(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        # calculate channel attention
        global_p1 = nuclear_pooling(en_ir)
        global_p2 = nuclear_pooling(en_vi)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * en_ir + global_p_w2 * en_vi
        return tensor_f


# sum of S V for each chanel
def nuclear_pooling(tensor):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, fs_type):
        super(Fusion_strategy, self).__init__()
        self.fs_type = fs_type
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()

    def forward(self, en_ir, en_vi):
        if self.fs_type =='add':
            fusion_operation = self.fusion_add
        elif self.fs_type == 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type == 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type == 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type == 'nuclear':
            fusion_operation = self.fusion_nuc

        f1_0 = fusion_operation(en_ir[0], en_vi[0])
        f2_0 = fusion_operation(en_ir[1], en_vi[1])
        f3_0 = fusion_operation(en_ir[2], en_vi[2])
        f4_0 = fusion_operation(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]

# class ConvLayer11(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(ConvLayer11, self).__init__()
#         reflection_padding = int(np.floor(kernel_size / 2))
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#         self.dropout = nn.Dropout2d(p=0.5)
#         self.norm=nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         out= self.norm(out)
#         out = F.relu(out, inplace=True)
#
#         return out
class DCBAM(nn.Module):
    def __init__(self,channel):
        super(DCBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        self.conv = GhostModule(inp=3 * channel, oup=channel, dw_size=1, stride=1)
#         # self.conv1 = GhostModule(inp=2 * channel, oup=channel, dw_size=1, stride=1)
#
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = out + x
        out1 = self.spatial_attention(x) * x
        out1 = out1 + x
#         #
        o1 = torch.max(out, out1)
        o2 = torch.add(out, out1)
        o3 = torch.mul(out, out1)
        out = torch.cat([o1, o2, o3], dim=1)
        out = self.conv(out)
#         # out=torch.cat([out,x],dim=1)
#         # out=self.conv1(out)
#         # out = out+x+out1
#
#
        # out = self.channel_attention(x) * x
        # out = self.spatial_attention(out) * out
        # out = x + out
        return out
# class DCBAM(nn.Module):
#     def __init__(self,channel):
#         super(DCBAM, self).__init__()
#
#         self.conv5 = GhostModule(inp=channel, oup=channel, dw_size=5, stride=1)
#         self.conv3_1 = GhostModule(inp=channel, oup=channel, dw_size=3, stride=1)
#
#         self.conv3_2 = GhostModule(inp=channel, oup=channel, dw_size=3, stride=1)
#         self.conv1 = GhostModule(inp=channel, oup=channel, dw_size=1, stride=1)
#
#
#     def forward(self,x):
#         # x00 = self.conv3_2(x) + x
#         x00=self.conv3_2(x)+x
#
#         x_5=self.conv5(x00)
#         x_3=self.conv3_1(x_5)
#         x_1=self.conv1(x_3)
#         x0=x_1+x_3+x_5+x00
#
#         p=torch.mul(x,x0)+x
#         return p



# NestFuse network - light, no desnse
class NestFuse_light2_nodense(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1):
        super(NestFuse_light2_nodense, self).__init__()
        # self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        self.CMCSA1 = DCBAM(channel=112)
        self.CMCSA2 = DCBAM(channel=160)
        self.CMCSA3 = DCBAM(channel=208)


        # encoder
        # [112, 160, 208, 256]
        self.conv0 =GhostModule(input_nc, output_filter, dw_size=1, stride=stride)
        # self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        # # no short connection
        # self.DB1_2 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        # self.DB2_2 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        # self.DB1_3 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2+ nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        # if self.deepsupervision:
        #     self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        #     self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        #     self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        #     # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        # else:
        self.conv_out = GhostModule(nb_filter[0], output_nc, dw_size=1, stride=stride)
        # self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x1_0=self.CMCSA1(x1_0)+x1_0

        x2_0 = self.DB2_0(self.pool(x1_0))
        x2_0 = self.CMCSA2(x2_0) + x2_0

        x3_0 = self.DB3_0(self.pool(x2_0))
        x3_0 = self.CMCSA3(x3_0) + x3_0

        x4_0 = self.DB4_0(self.pool(x3_0))
        # x2_0 = self.DB2_0(self.pool(x1_0))
        # x3_0 = self.DB3_0(self.pool(x2_0))
        # x4_0 = self.DB4_0(self.pool(x3_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def decoder(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        # if self.deepsupervision:
        #     output1 = self.conv1(x1_1)
        #     output2 = self.conv2(x1_2)
        #     output3 = self.conv3(x1_3)
        #     # output4 = self.conv4(x1_4)
        #     return [output1, output2, output3]
        # else:
        output = self.conv_out(x1_3)
        return output


    def decoder_eval(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        # if self.deepsupervision:
        #     output1 = self.conv1(x1_1)
        #     output2 = self.conv2(x1_2)
        #     output3 = self.conv3(x1_3)
        #     # output4 = self.conv4(x1_4)
        #     return [output1, output2, output3]
        # else:
        output = self.conv_out(x1_3)
        return output
# 564138 2.256552M
#2154888 8.619552M
nb_filter = [112,160, 208,256]
fs_type = 'res'
nC = 16
# input1 = torch.randn(channel, channel, 64, 64)
model=Fusion_network(nb_filter,fs_type)

# model=model(input1,input2)
para = sum([np.prod(list(p.size())) for p in model.parameters()])
print(para)
type_size = 4
print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))


# channel=16
# input1 = torch.randn(channel, channel, 64, 64)
# input2 = torch.randn(16, 16, 64, 64)
# model= CBE(channel=16)
# model=model(input1)
# print(model.shape)
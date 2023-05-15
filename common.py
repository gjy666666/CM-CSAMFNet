import torch
from torch import nn

class relect_conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=4,stride=2,pad=1):
        super(relect_conv, self).__init__()
        self.conv=nn.Sequential(
            nn.ReplicationPad2d(pad),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,
                      padding=0)
        )
    def forward(self,x):
        out=self.conv(x)
        return out


class gradient(nn.Module):
    def __init__(self,cha):
        super(gradient, self).__init__()
        self.channel = cha

    def forward(self,input):
        """
        求图像梯度, sobel算子
        :param input:
        :return:
        """
        input=input.cuda()
        # filter1 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel,kernel_size=3,  padding=1, stride=1).cuda()
        # filter2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3,  padding=1, stride=1).cuda()
        filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1).cuda()
        filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1).cuda()
        # GX
        filter1.weight.data = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).reshape(1, 1, 3, 3).expand(self.channel,self.channel,3,3).cuda()
        # GY
        filter2.weight.data = torch.tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
        ]).reshape(1, 1, 3, 3).expand(self.channel,self.channel,3,3).cuda()

        g1 = filter1(input)
        g2 = filter2(input)
        # 得到梯度
        image_gradient = torch.abs(g1) + torch.abs(g2)
        return image_gradient

class laplacian(nn.Module):
    def __init__(self,channel):
        super(laplacian, self).__init__()
        self.channel = channel
    def forward(self,input):
        filter1 = nn.Conv2d(kernel_size=3, in_channels=self.channel, out_channels=self.channel, bias=False, padding=1, stride=1)
        filter1.weight.data= torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ]).reshape(1, 1, 3, 3).expand(self.channel,self.channel,3,3).cuda()
        # print(filter1.weight.data.shape)
        # filter1.weight.data = p1.repeat(10,1)
        # print(filter1.weight.data.shape())
        # print(filter1_kernel.shape())

        g1 = filter1(input)
        image_gradient=torch.abs(g1)
        return image_gradient
#
# def clamp(value, min=0., max=1.0):
#     """
#     将像素值强制约束在[0,1], 以免出现异常斑点
#     :param value:
#     :param min:
#     :param max:
#     :return:
#     """
#     return torch.clamp(value, min=min, max=max)
#
#
# def RGB2YCrCb(rgb_image):
#     """
#     将RGB格式转换为YCrCb格式
#     :param rgb_image: RGB格式的图像数据
#     :return: Y, Cr, Cb
#     """
#
#     R = rgb_image[0:1]
#     G = rgb_image[1:2]
#     B = rgb_image[2:3]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#     Cr = (R - Y) * 0.713 + 0.5
#     Cb = (B - Y) * 0.564 + 0.5
#
#     Y = clamp(Y)
#     Cr = clamp(Cr)
#     Cb = clamp(Cb)
#     return Y, Cb, Cr
#
#
# def YCrCb2RGB(Y, Cb, Cr):
#     """
#     将YcrCb格式转换为RGB格式
#     :param Y:
#     :param Cb:
#     :param Cr:
#     :return:
#     """
#     ycrcb = torch.cat([Y, Cr, Cb], dim=0)
#     C, W, H = ycrcb.shape
#     im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
#     mat = torch.tensor(
#         [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
#     ).to(Y.device)
#     bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
#     temp = (im_flat + bias).mm(mat)
#     out = temp.transpose(0, 1).reshape(C, W, H)
#     out = clamp(out)
#     return out
#
#
#
# def TV_Loss(IA,IF):
#     r=IA-IF
#     batch_size=r.shape[0]
#     h=r.shape[2]
#     w=r.shape[3]
#     tv1=torch.pow((r[:,:,1:,:]-r[:,:,:h-1,:]),2).mean()
#     tv2=torch.pow((r[:,:,:,1:]-r[:,:,:,:w-1]),2).mean()
#     return tv1+tv2
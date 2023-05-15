# from __future__ import print_function
import argparse
import time

import numpy

# from network1 import Fusion_strategy
from load_model1 import load_model1, load_model2, load_model3
import imageio
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch jun')
parser.add_argument('--test_folder', type=str, default='./test', help='input image to use')
parser.add_argument('--model', type=str, default='./model/model.pth', help='model file to use')
parser.add_argument('--save_folder', type=str, default='./test', help='input image to use')

parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda', default='true')
opt = parser.parse_args()
print(opt)


def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img


def main():
    start_time = time.time()
    # # 载入MS-AE网络
    # path = 'model/AE/AE.model'
    path = 'model/AE.model'
    # path = 'model/AE/0.93AE.model'
    # path = 'model/Para/beta/0.93AE.model'
    # path = 'model/Para/0.5-single.model'
    model = load_model1(path)
    # print(model)
    for num in range(3):
        index = num +1
        # images_list1 = 'images1/MRI-SPECT/MRI/' + str(index) + '.png'
        # images_list2 = 'images1/MRI-SPECT/SPECT/' + str(index) + '.png'
        images_list1 = 'images1/MRI-PET/MRI/' + str(index) + '.png'
        images_list2 = 'images1/MRI-PET/PET/' + str(index) + '.png'
        img1 = Image.open(images_list1).convert('L')
        img0 = Image.open(images_list2).convert('YCbCr')
        y1 = img1
        y0, cb0, cr0 = img0.split()
        LR1 = y1
        LR0 = y0
        LR1 = Variable(ToTensor()(LR1)).view(1, -1, LR1.size[1], LR1.size[0])
        LR0 = Variable(ToTensor()(LR0)).view(1, -1, LR0.size[1], LR0.size[0])
        # print(LR0.shape)
        # print(LR1.shape)
        if opt.cuda:
            LR1 = LR1.cuda()
            LR0 = LR0.cuda()
        with torch.no_grad():
            # 灰度图融合
            tem1 = model.encoder(LR1)
            tem0 = model.encoder(LR0)
            print(type(tem1))
            print(type(tem0))
            # fs_type="max"
            # fusion_strategy = Fusion_strategy(fs_type)
            # tem=fusion_strategy(tem1,tem0)

            #
            # path_fusion = 'model/SPECT.model'
            path_fusion = 'model/PET.model'
            fusion_model = load_model3(path_fusion)
            tem = fusion_model(tem1, tem0)

            tem = model.decoder_eval(tem)
            tem = tem.cpu()
            # print(tem.shape())
            # cb0=numpy.array(cb0)
            # print(cb0.shape)
            # cb0=Image.fromarray(cb0)
            #
            # cr0 = numpy.array(cr0)
            # cr0 = Image.fromarray(cr0)

            # tem = tem.squeeze().squeeze()
            # tem = tem.numpy()
            # tem = tem * 255
            tem = process(tem, cb0, cr0)
            # print(type(tem))
            # tem=tem.cpu()
            # path = 'output/PET_our/test/our-PET' + str(index) + '.png'
            # path = 'output/SPECT-' + str(index) + '.png'
            path = 'output/PET-' + str(index) + '.png'
            imageio.imsave(path, tem)
    end_time = time.time()
    run_time=end_time-start_time
    print(run_time)
if __name__ == '__main__':
    main()

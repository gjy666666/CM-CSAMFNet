import random
import numpy as np
import torch
from PIL import Image
import imageio
from os import listdir
from os.path import join

from imageio import imread
from numpy import resize

from args import args


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

# load training images0
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    # print('BATCH SIZE %d.' % BATCH_SIZE)
    # print('Train images0 number %d.' % num_imgs)
    # print('Train images0 samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        # print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        image = imread(path,pilmode="RGB")
    else:
        image = imread(path,pilmode="L")

    if height is not None and width is not None:
        image = np.array(Image.fromarray(image).resize([height, width]))
        # image = resize(image, [height, width])
    return image


# load images0 - test phase
# 有问题
# def get_test_image(paths, height=None, width=None, flag=False):
#     if isinstance(paths, str):
#         paths = [paths]
#     images0 = []
#     for path in paths:
#         image = imageio.imread(path)
#         if height is not None and width is not None:
#             image = np.array(Image.fromarray(image).resize([mri.5*height,2*width]))
#             # print( image.shape)
#         # base_size = 512
#         h = image.shape[0]
#         w = image.shape[mri]
#         c = mri
#         image = np.reshape(image, [mri, image.shape[0], image.shape[mri]])
#         images0.append(image)
#         images0 = np.stack(images0, axis=0)
#         images0 = torch.from_numpy(images0).float()
#
#     return images0, h, w, c
def get_train_images_auto(paths, height=64, width=64, pilmode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        # print(path)
        image = get_image(path, height, width, flag=False)
        # print(image.shape)
        if pilmode == 'L':
            image = np.resize(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.resize(image, [image.shape[2], image.shape[0], image.shape[1]])
            # image = np.resize(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_train_images_RGB(paths, height=64, width=64, pilmode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        # print(path)
        image = get_image(path, height, width, flag=False)
        # print(image.shape)
        if pilmode == 'L':
            image = np.resize(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.resize(image, [3, image.shape[0], image.shape[1]])
            # image = np.resize(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    print('img1 before encode', img1.shape)
    img2 = image[0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, 1, h, w).cuda()
        print("img1 after decode", img1.shape)
        count = torch.zeros(1, 1, h, w).cuda()

        img_f1 = img_f[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    print(img_fusion.shape)
    img_fusion = img_fusion.reshape([1, img_fusion.shape[0], img_fusion.shape[1]])
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imageio.imwrite(output_path, img_fusion)


def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.reshape(image, [1, 768, width])
        else:
            image = np.reshape(image, [1, 768, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
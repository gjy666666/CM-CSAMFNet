
import os
from math import exp
import numpy as np

from net1 import Fusion_network
import utils
from load_model1 import load_model1
import time
from tqdm import trange
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import pytorch_msssim
EPSILON = 1e-5
from args import args

def hxx(x, y):
    size = x.shape[-1]
    # print(size)
    px = np.histogram(x, 256, (0, 255))[0] / size
    py = np.histogram(y, 256, (0, 255))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    r = hx + hy - hxy
    return r

def main():
    original_imgs_path = utils.list_images(args.dataset_vi)

    train_num =args.train_num
    # train_num = 1000
    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)
    alpha_list = [0]

    w_all_list = [[2,0.03]]
    for w_w in w_all_list:
        w1, w2 = w_w
        for alpha in alpha_list:
            train(original_imgs_path, alpha, w1, w2)




def train(original_imgs_path, alpha, w1, w2):
    batch_size = args.batch_size
    # load network model

    nb_filter = [112,160, 208,256]
    f_type = 'res'
    with torch.no_grad():
        path = 'AE.model'
        # path = 'model/AE/0.93AE.model'
        # path = '/Share/home/Z21301084/test/our/model/0.93AE.model'
        # path = '/Share/home/Z21301084/test/our/model/0.83AE_single.model'
        # path = 'model/AE/AE.model'
        # path ='model/Para/0.5-single.model'
        # path = 'Relu/AE50_SPECT.model'
        nest_model = load_model1(path)
        nest_model.cuda()
        nest_model.eval()

        # cls_model = Illumination_classifier()
        # cls_model.load_state_dict(torch.load(args.cls_pretrained))
        # cls_model.cuda()
        # cls_model.eval()
        fusion_model = Fusion_network(nb_filter, f_type)
        # print(fusion_model)
        fusion_model.train()
        fusion_model.eval()
        fusion_model.cuda()

    if args.resume_fusion_model is not None:
        print('Resuming, initializing layer net using weight from {}.'.format(args.resume_fusion_model))
        fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
    # optimizer = SGD(fusion_model.parameters(), args.lr)
    optimizer = Adam(fusion_model.parameters(), args.lr)
    # print(args.lr)
    # MSE_fun = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss(reduction="mean")
    mse_loss = torch.nn.MSELoss()

    # ssim_loss = pytorch_msssim.ssim
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        nest_model.cuda()
        fusion_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    # temp_path_model = os.path.join(args.save_fusion_model)
    # temp_path_loss = os.path.join(args.save_loss_dir)
    # if os.path.exists(temp_path_model) is False:
    #     os.mkdir(temp_path_model)
    #
    # if os.path.exists(temp_path_loss) is False:
    #     os.mkdir(temp_path_loss)

    # temp_path_model_w = os.path.join(args.save_fusion_model, str(alpha))
    # temp_path_loss_w  = os.path.join(args.save_loss_dir, str(alpha))
    # a = 'test'
    # temp_path_model_w = os.path.join(args.save_fusion_model, str(a))
    # temp_path_loss_w = os.path.join(args.save_loss_dir, str(a))
    #
    # if os.path.exists(temp_path_model_w) is False:
    #     os.mkdir(temp_path_model_w)
    #
    # if os.path.exists(temp_path_loss_w) is False:
    #     os.mkdir(temp_path_loss_w)

    Loss_feature = []
    Loss_ssim = []

    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_fea_loss = 0.

    for e in tbar:
        print('Epoch %d.....' % e)
        print(e)
        image_set_vi, batches = utils.load_dataset(original_imgs_path, batch_size)

        path_ir=utils.list_images(args.dataset_ir)
        path_ir= path_ir[:args.train_num]
        random.shuffle(path_ir)
        image_set_ir, batches = utils.load_dataset(path_ir, batch_size)
        # random.shuffle(original_imgs_path)
        # print(batches)
        nest_model.cuda()
        fusion_model.cuda()

        count = 0

        # batches:迭代总轮数
        for batch in range(batches):
            # YSPECT
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
            # img_vi_RGB = utils.get_train_images_RGB(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, pilmode='RGB')
            img_vi = utils.get_train_images_RGB(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, pilmode='L')
            # img_vi0=img_vi.numpy()
            # img_vi0=np.reshape(img_vi0, -1)
            # print(type(img_vi0))


            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            # image_paths_ir = [x.replace('visible', 'Inf') for x in image_paths_vi]
            img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, pilmode='L')
            # img_ir0 = img_ir.numpy()
            # img_ir0 = np.reshape(img_ir0, -1)
            # print(type(img_ir0))

            count += 1
            optimizer.zero_grad()

            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)
            # img_vi_RGB = Variable(img_vi_RGB, requires_grad=False)

            # if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
                # img_vi_RGB = img_vi_RGB.cuda()
            # get layer image
            # encoder
            en_ir = nest_model.encoder(img_ir)
            en_vi = nest_model.encoder(img_vi)
            # fusion_model
            f = fusion_model(en_ir, en_vi)

             # decoder
            output = nest_model.decoder_eval(f)
            output0 = output.cpu().detach().numpy()
            output0 = np.reshape(output0, -1)
            # print(type(output0))
            # print(output.shape)

            x_ir = Variable(img_ir.data.clone(), requires_grad=False)
            x_vi = Variable(img_vi.data.clone(), requires_grad=False)
            # x_vi_RGB= Variable(img_vi_RGB.data.clone(), requires_grad=False)

            loss1_value = 0.
            loss2_value = 0.
            loss3_value= 0.
            # Y分量
            # for output in outputs:
            output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
            output = output * 255

            # ssim_loss_temp1 = ssim_loss(output, x_ir)
            # ssim_loss_temp2 = ssim_loss(output, x_vi)
            # ssim_loss_temp1 = ssim_loss(output, x_ir, normalize=True)
            # MRI image
            ssim_loss_temp2 = ssim_loss(output, x_vi, normalize=True)

             #
            # M1=hxx(output0, img_ir0)
            # M2 = hxx(output0, img_vi0)
            # gama=exp(M1)/(exp(M1)+exp(M2))
            # gama=1
            # # print(gama)
            # # 1-ssim_loss_temp1:功能信息
            # # 1-ssim_loss_temp2:细节信息
            #
            # # ssim_loss1 = (1 - gama) * (1 - ssim_loss_temp1)
            # ssim_loss2 = gama * (1 - ssim_loss_temp2)
            #
            # # ssim_loss2 =  (1 - ssim_loss_temp2)
            # # ssim_loss1 =  gama *(1 - ssim_loss_temp1)
            # # ssim_loss2 =  gama*(1 - ssim_loss_temp2)
            #
            # ssim_loss_all = ssim_loss2
            loss1_value += alpha * (1 - ssim_loss_temp2)

            # 亮度损失
            # pred = cls_model(x_vi_RGB)
            # day_p = pred[:, 0]
            # night_p = pred[:, 1]
            # W_vis = day_p / (day_p + night_p)
            # W_ir = 1 - W_vis
            # W_vis= W_vis[:, None, None, None]
             # W_ir = W_ir[:, None, None, None]
            # print( W_vis)
            # print(type( W_vis))

            g2_ir_fea = en_ir
            g2_vi_fea = en_vi
            g2_fuse_fea = f
            w_ir = [w1, w1, w1, w1]
            w_vi = [w2, w2, w2, w2]
            # print( w_ir[0])
            # print(type(w_ir[0]))
            # W_ir=W_ir.int()
            # W_vis=W_vis.int()
            # w_ir = [W_ir,W_ir, W_ir, W_ir]
            # w_vi = [W_vis, W_vis, W_vis, W_vis]

            #
            F1 = L1_loss(g2_fuse_fea[0], w_ir[0] * g2_ir_fea[0] + w_vi[0] * g2_vi_fea[0])
            F2 = L1_loss(g2_fuse_fea[1], w_ir[1] * g2_ir_fea[1] + w_vi[1] * g2_vi_fea[1])
            F3 = L1_loss(g2_fuse_fea[2], w_ir[2] * g2_ir_fea[2] + w_vi[2] * g2_vi_fea[2])
            F4 = L1_loss(g2_fuse_fea[3], w_ir[3] * g2_ir_fea[3] + w_vi[3] * g2_vi_fea[3])

            # F1 = L1_loss(g2_fuse_fea[0], g2_ir_fea[0])+L1_loss(g2_fuse_fea[0], g2_vi_fea[0])
            # F2 = L1_loss(g2_fuse_fea[1], g2_ir_fea[1])+L1_loss(g2_fuse_fea[1], g2_vi_fea[1])
            # F3 = L1_loss(g2_fuse_fea[2], g2_ir_fea[2])+L1_loss(g2_fuse_fea[2], g2_vi_fea[2])
            # F4 = L1_loss(g2_fuse_fea[3], g2_ir_fea[3])+L1_loss(g2_fuse_fea[3], g2_vi_fea[3])

            eps = 1e-5
            p1 = torch.exp(F1) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)
            p2 = torch.exp(F2) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)
            p3 = torch.exp(F3) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)
            p4 = torch.exp(F4) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)

            for ii in range(4):
                # w_fea = [1, 4, 6, 10]
                w_fea = [p1*100, p2*100, p3*100, p4*100]
                # w_fea = [100, 100, 100, 100]
                g2_ir_temp = g2_ir_fea[ii]
                g2_vi_temp = g2_vi_fea[ii]
                g2_fuse_temp = g2_fuse_fea[ii]

                # fea_loss = w_fea[ii] * (w_ir[ii] * L1_loss(g2_fuse_temp, g2_ir_temp) + w_vi[ii] * L1_loss(g2_fuse_temp, g2_vi_temp))
                fea_loss = w_fea[ii] * L1_loss(g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp)
                # fea_loss = w_fea[ii] * mse_loss(g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp)
                loss2_value += fea_loss

            loss1_value /= len(output)
            loss2_value /= len(output)
                # loss3_value /= len(output)
            total_loss = loss1_value + loss2_value
            total_loss.backward()
            optimizer.step()

            all_fea_loss += loss2_value.item()
            all_ssim_loss += loss1_value.item()
            # all_context_loss += loss3_value.item()
            # all_color_loss += loss3_value.item()
            # 训练过程种颜色偏差严重Lcolor

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\t Alpha: {} \tW-Inf: {}\tEpoch {}:\t[{}/{}]\t detail loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), alpha, w1, e + 1, count, batches,
                                             all_ssim_loss / args.log_interval,
                                             all_fea_loss / args.log_interval,
                                             # all_context_loss / args.log_interval,
                                             (all_fea_loss + all_ssim_loss) / args.log_interval
                )
                print(e)
                tbar.set_description(mesg)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_feature.append(all_fea_loss / args.log_interval)
                # Loss_context.append(all_context_loss / args.log_interval)
                Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
                count_loss = count_loss + 1
                all_ssim_loss = 0.
                all_fea_loss = 0.
                    # all_context_loss=0.

            # if (batch + 1) % (200000 * args.log_interval) == 0:
            #
            # 	# save model
            # 	fusion_model.eval()
            # 	fusion_model.cuda()
            # 	save_model_filename = "RGB_Epoch_" + str(e) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".model"
            # 	save_model_path = os.path.join(temp_path_model, save_model_filename)
            # 	torch.save(fusion_model.state_dict(), save_model_path)
            #
            # 	# save loss YSPECT1
             # 	# -----------------------------------------------------
            # 	# pixel loss
            # 	loss_data_ssim = Loss_ssim
            # 	loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
            # 	scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
            #
            # 	# SSIM loss
            # 	loss_data_fea = Loss_feature
            # 	loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
            # 	scio.savemat(loss_filename_path, {'loss_fea': loss_data_fea})
            #
            # 	# grd loss
            # 	# loss_data_grd = Loss_grd
            # 	# loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(
            # 	# 	count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
            # 	# scio.savemat(loss_filename_path, {'loss_grd': loss_data_grd})
            #
            # 	# color loss
            # 	# loss_data_hist = Loss_hist
            # 	# loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(
            # 	# 	count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
             # 	# scio.savemat(loss_filename_path, {'loss_hist': loss_data_hist})
            #
            # 	# color loss
            # 	# loss_data_mse = Loss_mse
            # 	# loss_filename_path = temp_path_loss_w + "/loss_mse_epoch_" + str(args.epochs) + "_iters_" + str(
            # 	# 	count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
            # 	# scio.savemat(loss_filename_path, {'loss_mse': loss_data_mse})
            #
            # 	# all loss
            # 	loss_data = Loss_all
            # 	loss_filename_path = temp_path_loss_w + "/loss_all_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
            # 	scio.savemat(loss_filename_path, {'loss_all': loss_data})
            #
            # 	# fontP = FontProperties()
            # 	# fontP.set_size('large')
            # 	# plt.plot(loss_data, 'b', label='$Loss_all')
            # 	# plt.plot(loss_data_ssim, 'c', label='$ssim$')
            # 	# plt.plot(loss_data_fea, 'c', label='$loss_fea$')
            # 	# plt.xlabel('epoch', fontsize=15)
            # 	# plt.ylabel('Loss values', fontsize=15)
            # 	# plt.legend(loc=2, prop=fontP)
             # 	# # plt.title('FunFuseAn $\lambda = 0.8, \gamma_{ssim} = 0.5, \gamma_{l2} = 0.5$', fontsize='15')
            # 	# plt.savefig('./results/1oss.png')
            #
            # 	fusion_model.train()
            # 	fusion_model.cuda()
            # 	tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
            # 	print(Loss_ssim)
            # 	# writer = SummaryWriter("logs")
            # 	# writer.add_scalar("ssim_loss", Loss_ssim, batch)
            # 	# writer.add_scalar("fea_loss", Loss_feature, batch)
            # 	# writer.add_scalar("all_loss", Loss_all, batch)

        # 五种 loss
        # loss_data_ssim = Loss_ssim
        # loss_filename_path = temp_path_loss_w + "/Final_loss_ssim_epoch_" + str(
        #     args.epochs) + "_lamda_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + "-gama0.8.mat"
        # scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
        #
        # loss_data_fea = Loss_feature
        # loss_filename_path = temp_path_loss_w + "/Final_loss_fea_epoch_" + str(
        #     args.epochs) + "_lamda_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + "-gama0.8.mat"
         # scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})
        # # scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})
        #
        #
        # # Total loss
        # loss_data = Loss_all
        # loss_filename_path = temp_path_loss_w + "/Final_loss_all_epoch_" + str(
        #     args.epochs) + "_lamda_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + "-gama0.8.mat"
        # scio.savemat(loss_filename_path, {'final_loss_all': loss_data})

        # save model
        fusion_model.eval()
        fusion_model.cuda()
        # save_model_path = "model/Para/PET.model"
        save_model_path = "Pallel-SPECT.model"
        # save_model_path = "model/AE/sing-newFusion.model"
        # save_model_path = "/Share/home/Z21301084/test/our/model/sing-new.model"
        # save_model_path = os.path.join(temp_path_model_w, save_model_filename)
        torch.save(fusion_model.state_dict(), save_model_path)
        print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    main()
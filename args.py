class args():
    # training args
    epochs=4

    batch_size=8
    HEIGHT =64
    WIDTH =64
    downsample = ['stride', "avgpool", "maxpool"]
    # dataset_ir = "train_64/Inf"
    # dataset_vi = "train_64/visible"rgb
    # dataset = "/Share/home/Z21301084/test/RFN1/UN1/train_SPECT_64/SPECT"

    # dataset= "/Share/home/Z21301084/test/RFN1/UN1/train_SPECT/train_SPECT"
    # dataset = "train_SPECT/T1"
    # dataset_vi= "/Share/home/Z21301084/test/RFN1/UN1/trainSP_64/T1"
    # dataset_ir= "/Share/home/Z21301084/test/RFN1/UN1/trainSP_64/SPECT"
    #
    # dataset_vi= "/Share/home/Z21301084/test/our/SPECT/MRI"
    # dataset_ir= "/Share/home/Z21301084/test/our/SPECT/SPECT"

    # dataset_vi= "/Share/home/Z21301084/test/our/PET/MRI"
    # dataset_ir= "/Share/home/Z21301084/test/our/PET/PET"
    workers=1
    arch='fusion_model'

    # dataset_vi = "msrs_train/Inf"
    # dataset_ir = "msrs_train/Vis"
    # dataset_vi = "PET/MRI"
    # dataset_ir = "PET/PET"
    dataset_vi= "SPECT/MRI"
    dataset_ir= "SPECT/SPECT"
    train_num=20000
    # 26900
    # dataset_fusion_path = 'SPECT-MRI'
    # dataset_fusion_path='/Share/home/Z21301084/test/our/SPECT'
    # dataset_ir = "/Share/home/Z21301084/test/DIVFusion/traincom/Inf"
    # dataset_vi= "/Share/home/Z21301084/test/DIVFusion/traincom/visible"

    # dataset_ir = "/Share/home/Z21301084/test/RFN1/UN1/train_msrs/infrared"
    # dataset_vi = "/Share/home/Z21301084/test/RFN1/UN1/train_msrs/visible"

    # dataset= "/Share/home/Z21301084/test/RFN1/MMI/COCO-train2017"
    # dataset = "/Share/home/Z21301084/test/our/PET/MRI"
    # dataset = "/Share/home/Z21301084/test/our/SPECT/MRI"
    dataset = "SPECT/MRI"
    # dataset = "SPECT-MRI/MRI"
    # dataset = "COCO-train2017/COCO-train2017"
    # save_model_dir_encoder = "models/model"
    # save_loss_dir = "models/loss"
    start_epoch = 0
    cuda = 1
    ssim_weight = [1, 10, 100, 1000, 10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']
    grad_weight = [1, 10, 100, 1000, 10000]

    lr =10e-4  # "learning rate, default is 0.001"
    lr_light = 10e-4  # "learning rate, default is 0.001"
    log_interval = 10  # "number of images1 after which the training loss is logged, default is 500"
    resume = None
    # train_num=26900
    # for test, model_default is the model used in paper
    # model_default = './model/UNFusion.pth'
    # model_deepsuper = 'UNFusion.model'
    #
    # save_fusion_model_onestage="./onestage/model"
    # save_loss_dir_onestage="./onestage/loss"
    resume_fusion_model=None
    # save_fusion_model = "/Share/home/Z21301084/test/RFN1/UN1/model/function/"
    # save_loss_dir = '/Share/home/Z21301084/test/RFN1/UN1/models/loss/function/'

    # save_loss_dir = 'model/loss/'
    # save_fusion_model = "model/"
    #

    # save_fusion_model = "/Share/home/Z21301084/test/our/model/"
    # save_loss_dir = '/Share/home/Z21301084/test/our/model/loss/'

    cls_pretrained = "model/best_RGB.pth"
    # cls_pretrained="/Share/home/Z21301084/test/RFN1/UN1/model/best_cls.pth"
    # save_path = "/Share/home/Z21301084/test/RFN1/UN1/model"
    save_path = "/Share/home/Z21301084/test/our/model/"
    arch_cls = 'cls_model'
    dataset_train_path='/Share/home/Z21301084/test/RFN1/UN1/train_msrs'
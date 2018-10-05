import os
import sys
import argparse
import pandas as pd
import platform

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, RandomSampler

try:
    from model import unet, unetresnet, unetfrog
    from data import *
    from data.dataset import TGSSaltDataset
    from common.transform import *
    from common.loss import *
    from trainer import train
    from common.lovasz_softmax import *
    from cli import *
except ImportError:
    from .model import unet, unetresnet, unetfrog
    from .data import *
    from .data.dataset import TGSSaltDataset
    from .common.transform import *
    from .common.loss import *
    from .trainer import train
    from .common.lovasz_softmax import *
    from .cli import *


def train_augment(image, mask):

    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.125)
        if c == 1:
            image, mask = do_elastic_transform2(image, mask, grid=10,
                                                distort=np.random.uniform(0, 0.1))
        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1,
                                                 angle=np.random.uniform(0, 10))

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.05, +0.05))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.05, 1 + 0.05))
        if c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.05, 1 + 0.05))
        # if c==1:
        #     image = do_invert_intensity(image)

    image, mask = do_center_pad_to_factor2(image, mask, factor=32)
    return image, mask


def create_data_loaders(img_root_path, train_file_list, valid_file_list, train_transformation=None):

    train_dataset = TGSSaltDataset(root_path=img_root_path,
                                   file_list=train_file_list,
                                   type=args.image_type,
                                   mode="train",
                                   transform=train_transformation,
                                   )

    valid_dataset = TGSSaltDataset(root_path=img_root_path,
                                   file_list=valid_file_list,
                                   type=args.image_type,
                                   mode="train",
                                   )

    data_loader = {
        'train': DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        batch_size  = args.batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True,
                        ),
        'valid': DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = args.batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        ),
    }

    return data_loader


def main(img_root, model_params_path, load=False):
    #
    # DataLoader    -----------------------------------------------------------
    print("####################   Use fold - {}   ####################".format(args.fold))
    print("####################    Image ({})     ####################".format(args.image_type))

    if args.verification:
        print("**************** verification model:\n")
        train_df = pd.read_csv(os.path.join(code_root, "dataset_split", "train_verification.csv"))
        valid_df = pd.read_csv(os.path.join(code_root, "dataset_split", "valid_verification.csv"))
    else:
        train_df = pd.read_csv(os.path.join(code_root, "dataset_split", "train_fold_" + str(args.fold) + ".csv"))
        valid_df = pd.read_csv(os.path.join(code_root, "dataset_split", "valid_fold_" + str(args.fold) + ".csv"))

    train_file_list = list(train_df['id'].values)
    valid_file_list = list(valid_df['id'].values)

    data_loader = create_data_loaders(img_root, train_file_list, valid_file_list, train_transformation=train_augment)

    data_size = {
        'train': len(data_loader['train'].dataset),
        'valid': len(data_loader['valid'].dataset),
    }

    #
    # Model         -----------------------------------------------------------

    # model = unet.UNet(in_channel=1, n_classes=1)
    model = unetresnet.UNetResNet(encoder_depth=34, num_classes=1)
    # model = unetfrog.SaltNet()

    if torch.cuda.is_available():
        model = model.cuda()

    if args.load or load:
        model.load_state_dict(torch.load(model_params_path))

    
    if args.consistency_type == "BCELoss":
        print("Use BCE Loss")
        criterion = nn.BCELoss()
    elif args.consistency_type == "FocalLoss":
        print("Use Focal Loss")
        criterion = FocalLoss2d()
    else:
        assert False, args.consistency_type
    
    #criterion = FocalLoss2d()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    """
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    """

    #
    # train         -----------------------------------------------------------

    #train(model, data_loader, data_size, args.epochs, args.lr, optimizer, criterion, save='loss', fold=args.fold)

    try:
        train(model, data_loader, data_size, args.epochs, args.lr, optimizer, criterion,
              save='acc', fold=args.fold, image_type=args.image_type, loss_type=args.consistency_type)

    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":

    code_root = os.getcwd()

    args = parse_commandline_args()

    if args.image_type == "pad":
        img_root = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train"
    elif args.image_type == "resize":
        img_root = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train"
    
    if not os.path.exists("./model_params"):
        os.mkdir("./model_params")

    model_params_name = ""
    model_params_path = os.path.join(code_root, "model_params", model_params_name)

    main(img_root, model_params_path, load=False)

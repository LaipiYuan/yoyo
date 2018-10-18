import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import cv2
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

try:
    from common.transform import *
    from model import unet, unetresnet, unetfrog
    from common.run_length_encode import *
except ImportError:
    from .common.transform import *
    from .model import unet, unetresnet, unetfrog
    from .common.run_length_encode import *


def create_parser():
    parser = ArgumentParser(description="Salt Submit'")

    parser.add_argument('-d', '--depth', dest='depth', default=34, type=int,
                        help='depth of ResNet')
    parser.add_argument('-t', '--type', dest='image_type', default='pad', type=str,
                        choices=['pad', 'resize'],
                        help='image type pad or resize')

    parser.add_argument('--tta', dest='tta', default=False,
                        help='save tta probability')
    parser.add_argument('-s', '--submit', dest='submit', default=False,
                        help='call submit funcion')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


#
# TTA (Test Time Augmentation): flip horizontally -----------------------------
def test_augment_flip_rl(image, image_type):
    image = do_horizontal_flip(image)

    if image_type == "pad":
        image = do_center_pad_to_factor(image, factor=32)
    elif image_type == "resize":
        image = do_resize(image, resize=128)

    return image


def test_unaugment_flip_rl(prob, image_type):
    if image_type == "pad":
        dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
        prob = prob[:, dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
        prob = prob[:, :, ::-1]
    elif image_type == "resize":
        prob = do_resize(prob, resize=101)

    return prob


def test_augment_null(image, image_type):

    if image_type == "pad":
        image = do_center_pad_to_factor(image, factor=32)
    elif image_type == "resize":
        image = do_resize(image, resize=128)

    return image


def test_unaugment_null(prob, image_type):
    if image_type == "pad":
        dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
        prob = prob[:, dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    elif image_type == "resize":
        prob = do_resize(prob, resize=101)

    return prob


def do_resize(img, resize=128):
    img_resize = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    return img_resize


def hwc_to_chw(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        image = np.transpose(image, axes=[2, 0, 1])
    return image


def run_tta_predict(model, img_root, image_type, augment, fold):
    if augment == 'null':
        test_augment   = test_augment_null
        test_unaugment = test_unaugment_null
    if augment == 'flip_rl':
        test_augment   = test_augment_flip_rl
        test_unaugment = test_unaugment_flip_rl

    all_prob = []

    model = model.eval()

    N = len(list(os.listdir(img_root)))
    for index, name in enumerate(sorted(os.listdir(img_root))):
        if index % 1000 == 0:
            print('{}/{}'.format(index, N))

        image = cv2.imread(os.path.join(img_root, name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        image = test_augment(image, image_type)
        image = hwc_to_chw(image)

        image = torch.from_numpy(image).unsqueeze(0)

        if torch.cuda.is_available():
            image = Variable(image.cuda())
        else:
            image = Variable(image)

        with torch.no_grad():
            mask_pred = model(image)
            mask_prob = F.sigmoid(mask_pred).squeeze(0)

        prob = mask_prob.data.cpu().numpy()
        prob = test_unaugment(prob, image_type)
        prob = np.squeeze(prob, axis=0)
        all_prob.append(prob)

    #all_prob = np.concatenate(all_prob)
    all_prob = np.array(all_prob)
    all_prob = (all_prob * 255).astype(np.float32)
    np.save(os.path.join(code_root, 'ensemble', '%s_%s.prob.float32.npy' % (fold, augment)), all_prob)
    print(all_prob.shape)


def run_submit(augment, fold, threshold=0.5):
    if augment in ['null', 'flip_rl']:
        augmentation = [
            1, os.path.join(code_root, 'ensemble', '%s_%s.prob.float32.npy' % (fold, augment)),
        ]

    if augment == 'tta_ensemble':
        augmentation = [
            1, os.path.join(code_root, 'ensemble', '%s_%s.prob.float32.npy' % (fold, 'null')),
            1, os.path.join(code_root, 'ensemble', '%s_%s.prob.float32.npy' % (fold, 'flip_rl')),
        ]

    augmentation = np.array(augmentation, dtype=object).reshape(-1, 2)
    num_augments = len(augmentation)

    #
    # Ensemble      -----------------------------------------------------------
    w, augment_file = augmentation[0]
    all_prob = w * np.load(augment_file).astype(np.float32) / 255

    all_w = w
    for i in range(1, num_augments):
        w, augment_file = augmentation[i]
        prob = w * np.load(augment_file).astype(np.float32) / 255
        all_prob += prob
        all_w += w

    all_prob /= all_w
    all_prob = all_prob > threshold
    print(all_prob.shape)

    #
    # Submit        -----------------------------------------------------------
    id_ = []
    rle_mask = []

    N = len(list(os.listdir(test_img_root)))
    for index, name in enumerate(sorted(os.listdir(test_img_root))):
        if index % 1000 == 0:
            print('{}/{}'.format(index, N))

        id_.append(str(name)[:-4])

        if all_prob[index].sum() <= 0:
            encoding = ''
        else:
            encoding = rle_encode(all_prob[index])
        assert (encoding != [])

        rle_mask.append(encoding)

    df = pd.DataFrame({'id': id_ , 'rle_mask': rle_mask}).astype(str)
    df.to_csv(os.path.join("submit_file", 'submit_%s_%s_prob.csv' % (fold, augment)),
              index=False, columns=['id', 'rle_mask'])

    print("Submit finished")


def main(model_params_path):
    #
    # Model         -----------------------------------------------------------

    #model = unet.UNet(in_channel=1, n_classes=1)
    model = unetresnet.UNetResNet(encoder_depth=args.depth, num_classes=1)
    #model = unetfrog.SaltNet()

    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(model_params_path))
    else:
        model.cpu()
        model.load_state_dict(torch.load(model_params_path))

    if args.tta:
        for tta in ['null', 'flip_rl']:
            run_tta_predict(model, test_img_root, args.image_type, augment=tta, fold=model_params_name[:5])

    if args.submit:
        for tta in ['null', 'flip_rl', 'tta_ensemble']:
            run_submit(tta, fold=model_params_name[:5], threshold=0.546543706)


if __name__ == "__main__":
    IMAGE_HEIGHT = 101
    IMAGE_WIDTH  = 101

    args = parse_commandline_args()
    code_root = os.getcwd()

    if args.image_type == "pad":
        test_img_root = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/images/"
        train_img_root = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train/"
    elif args.image_type == "resize":
        test_img_root = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/test/"
        train_img_root = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train"

    # ***
    model_params_name = "fold1_pad_restnet152_BCELoss_e154_tacc0.9927_tls0.00123_vacc0.9816_vls0.00624_lr0.000129_lb788.pth"
    model_params_path = os.path.join(code_root, "model_params_041", model_params_name)

    main(model_params_path)


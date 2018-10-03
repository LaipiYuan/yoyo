import os
import numpy as np
import cv2
import platform
from PIL import Image

from torch.utils.data.sampler import *
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

import pydensecrf.densecrf as dcrf

try:
    from data.dataset import TGSSaltDataset
    from model import unet, unetresnet
    from data.run_length_encode import *
    from evaluate_iou import *
    from calculate_iou import *
except ImportError:
    from .data.dataset import TGSSaltDataset
    from .model import unet, unetresnet
    from .common.run_length_encode import *
    from .evaluate_iou import *
    from .calculate_iou import *

batch_size = 20


def do_resize(img, resize=128):
    img_resize = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    return img_resize


def hwc_to_chw(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        image = np.transpose(image, axes=[2, 0, 1])
    return image


def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


def predict_mask(net, img, out_threshold=0.5, use_dense_crf=False):
    height, width = 101, 101

    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    # image = hwc_to_chw(img)
    # image = torch.from_numpy(image).unsqueeze(0)
    image = img.unsqueeze(0)

    if torch.cuda.is_available():
        image = Variable(image.cuda())
    else:
        image = Variable(image)

    with torch.no_grad():
        mask_pred = net(image)
        mask_prob = F.sigmoid(mask_pred).squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(101),
                transforms.ToTensor()
            ]
        )

        # mask = tf(mask_prob.cpu())
        mask = mask_prob.data.cpu().numpy()[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
        # mask_np = mask.squeeze().cpu().numpy()
        mask_np = mask.squeeze()
        print("mask_np ", mask.shape)

        img_ = img.cpu().numpy()[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
        print(img_.shape)

    if use_dense_crf:
        mask_np = dense_crf(img_.astype(np.uint8), mask)

    return (mask_np > out_threshold).astype(int) * 255


def get_output_filename(input_file_name):
    output_file_name = []

    for f in input_file_name:
        pathsplit = os.path.splitext(f)
        output_file_name.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return output_file_name


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


# def evaluate_iou():


# -----------------------------------------------------------------------------


def create_data_loaders(img_root_path, file_list, transformation=None):
    print("----------  In create_data_loaders  ----------")

    test_dataset = TGSSaltDataset(root_path=img_root_path,
                                  file_list=file_list,
                                  mode="test",
                                  transform=transformation,
                                  )

    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=batch_size,
                             drop_last=False,
                             num_workers=8,
                             pin_memory=True,
                             )

    return test_loader


# un-finished
def run_predict(net, data_loader):
    all_prob = []

    for image, mask in data_loader:
        if torch.cuda.is_available():
            image = Variable(image.cuda())
        else:
            image = Variable(image)

        mask_pred = net(image)
        mask_prob = F.sigmoid(mask_pred)

        prob = mask_prob.squeeze().data.cpu().numpy()
        all_prob.append(prob)

    all_prob = np.concatenate(all_prob)
    all_prob = (all_prob * 255).astype(np.uint8)


if __name__ == "__main__":

    image_file = "911efbb175.png"  # 911efbb175, b012e9ebb0, 9ca84742ef

    model_file_name = "fold5_restnet34_e_params_e62_tls0.00282_vls0.00348_lr0.002681.pth"

    if platform.system() == "Linux":
        model_file_root = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"
        # input_file = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train/images/" + image_file
        input_file = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train/images/" + image_file

    elif platform.system() == "Darwin":
        model_file_root = "."
        input_file = "/Users/liapck/DeepLearning/Unet/dataset/images" + image_file

    model_file = os.path.join(model_file_root, model_file_name)

    # load parameter
    # net = unet.UNet(in_channel=1, n_classes=1)
    # net = unetresnet.UNetResNet(encoder_depth=152, num_classes=1)
    net = unetresnet.UNetResNet(encoder_depth=34, num_classes=1)

    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(model_file))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model_file))

    # predict
    # img = Image.open(input_file)
    # img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    # img = cv2.imread(input_file).astype(np.float32) / 255
    img = load_image(input_file)

    mask = predict_mask(net, img, use_dense_crf=False)

    result = mask_to_image(mask)
    result.save(image_file)



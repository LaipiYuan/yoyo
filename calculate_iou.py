import os
import numpy as np
import pandas as pd
import cv2
import platform

from torch.utils.data.sampler import *
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
try:
    from model import unet, unetresnet, unetfrog
except ImportError:
    from .model import unet, unetresnet, unetfrog


def do_resize(img, resize=128):
    img_resize = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    return img_resize


def hwc_to_chw(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        image = np.transpose(image, axes=[2, 0, 1])
    return image


def load_image(path, mask=False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    height, width = img.shape

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

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT, value=0)

    img = hwc_to_chw(img)

    if mask:
        # Convert mask to 0 and 1 format
        # img = img[:, :, 0:1] // 255
        img = img // 255
        return torch.from_numpy(img).float()
    else:
        img = img / 255.0
        return torch.from_numpy(img).float()


def predict(img_path, mask_path, net):
    """Used for Kaggle submission: predicts and encode all test images"""
    id_ = []
    pred_num = []
    mask_num = []
    inter = []
    union = []
    threshold = []

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

    N = len(list(os.listdir(img_path)))
    for index, name in enumerate(os.listdir(img_path)):
        if index % 500 == 0:
            print('{}/{}'.format(index, N))

        id_.append(str(name)[:-4])

        # img = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        img = load_image(img_path + name)
        # img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        mask_true = cv2.imread(mask_path + name, cv2.IMREAD_GRAYSCALE).astype(np.float32) // 255

        # image = hwc_to_chw(img)
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

            mask = mask_prob.cpu().numpy()[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
            # mask = tf(mask_prob.cpu())
            # mask = tf(mask_pred.cpu().squeeze(0))
            mask_pred_np = mask.squeeze() > 0.4332206261113065  # 0.663294217 #0.5465437

        mask_true_np = np.array(mask_true)

        pred_num.append(mask_pred_np.sum())
        mask_num.append(mask_true_np.sum())
        inter_ = ((mask_pred_np == 1) & (mask_true_np == 1)).sum()
        union_ = ((mask_pred_np == 1) | (mask_true_np == 1)).sum()
        inter.append(inter_)
        union.append(union_)
        threshold.append(inter_ / union_)

    data = pd.DataFrame(data={"id": id_,
                              "pred": pred_num,
                              "mask": mask_num,
                              "inter": inter,
                              "union": union,
                              "threshold": threshold})

    data.to_csv("calculate_fold1_PAD_e145.csv", index=False)


if __name__ == "__main__":

    img_root_path = "."
    code_root_path = "."

    model_file_name = "fold1_restnet34_e_params_e145_tls0.00124_vls0.00318_lr0.001164.pth"

    if platform.system() == "Linux":
        model_file_root = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"
        # img_root_path = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train"
        img_root_path = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train/"
        code_root_path = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"

    train_image_path = img_root_path + "images/"
    train_mask_path = img_root_path + "masks/"

    model_file = os.path.join(model_file_root, model_file_name)

    net = unetresnet.UNetResNet(encoder_depth=34, num_classes=1)

    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(model_file))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model_file))

    predict(train_image_path, train_mask_path, net)
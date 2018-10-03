import os
import pandas as pd
import platform

from torch.utils.data.sampler import *

try:
    from data.dataset import TGSSaltDataset
    from model import unet, unetresnet, unetfrog
    from data.run_length_encode import *
    from predict import predict_mask
    from calculate_iou import *
except ImportError:
    from .data.dataset import TGSSaltDataset
    from .model import unet, unetresnet, unetfrog
    from .common.run_length_encode import *
    from .predict import predict_mask
    from .calculate_iou import *


def rle_encode(img, order='F'):
    """
    img: np.array: 1 - mask, 0 - background
    Returns
    -------
    run-length string of pairs of (start, length)
    """
    pixels = img.reshape(img.shape[0] * img.shape[1], order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle if rle else float('nan')


# Frog
def do_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    # if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
    #    rle[-2] = rle[-2] -1

    rle = ' '.join([str(r) for r in rle])
    return rle


def submit(net, path, csv_file):
    """Used for Kaggle submission: predicts and encode all test images"""
    id_ = []
    rle_mask = []
    N = len(list(os.listdir(path)))
    for index, name in enumerate(os.listdir(path)):
        if index % 500 == 0:
            print('{}/{}'.format(index, N))

        id_.append(str(name)[:-4])

        # image = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        # image = cv2.imread(path + name, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        image = load_image(path + name)

        mask = predict_mask(net, image, out_threshold=0.5465437)

        enc = rle_encode(mask)
        # enc = do_length_encode(mask)

        # rle_mask.append(' '.join(map(str, enc)))
        rle_mask.append(enc)

    df = pd.DataFrame({'id': id_, 'rle_mask': rle_mask}).astype(str)
    df.to_csv("submit_" + csv_file, index=False, columns=['id', 'rle_mask'])


if __name__ == '__main__':

    file = "fold5_restnet34_e_params_e62_tls0.00282_vls0.00348_lr0.002681.pth"

    if platform.system() == "Linux":
        model_file_root = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"
        # test_image_path = '/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/test/'
        test_image_path = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/images/"

    elif platform.system() == "Darwin":
        model_file_root = "."
        test_image_path = "."

    model_file = os.path.join(model_file_root, file)

    # load parameter
    # net = unet.UNet(in_channel=1, n_classes=1)
    # net = unetresnet.UNetResNet(encoder_depth=152, num_classes=1)
    net = unetresnet.UNetResNet(encoder_depth=34, num_classes=1)
    # net = unetfrog.SaltNet()

    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(model_file))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model_file))

    submit(net, test_image_path, file)

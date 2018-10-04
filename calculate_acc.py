import os
from PIL import Image
import platform
import cv2
import pydensecrf.densecrf as dcrf
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

try:
    from data.dataset import TGSSaltDataset
    from model import unet, unetresnet, unetfrog
    from data.run_length_encode import *
    from predict import predict_mask
    from calculate_iou import *
except ImportError:
    from .data.dataset import TGSSaltDataset
    from .model import unet, unetresnet, unetfrog
    from .data.run_length_encode import *
    from .predict import predict_mask
    from .calculate_iou import *



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

    #image = hwc_to_chw(img)
    #image = torch.from_numpy(image).unsqueeze(0)
    image = img.unsqueeze(0)

    if torch.cuda.is_available():
        image = Variable(image.cuda())
    else:
        image = Variable(image)

    with torch.no_grad():
        mask_pred = net(image)
        mask_prob = F.sigmoid(mask_pred).squeeze(0)

        #mask = tf(mask_prob.cpu())
        mask = mask_prob.data.cpu().numpy()[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
        mask_prob_flat = mask.reshape(-1)

    return mask_prob_flat


def compute_global_accuracy(pred, label):
    pred = (pred > 0.5).astype(int).tolist()
    label = label.tolist()

    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def calculate_acc(net, path, csv_file):
    id_ = []
    acc_list = []

    train_path = os.path.join(path, "images")
    mask_path = os.path.join(path, "masks")
    N = len(list(os.listdir(train_path)))
    for index, name in enumerate(os.listdir(train_path)):
        if index % 200 == 0:
            print('{}/{}'.format(index, N))

        id_.append(str(name)[:-4])

        #image = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        #image = cv2.imread(path + name, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        image = load_image(os.path.join(train_path, name))

        pred_mask_flat = predict_mask(net, image, out_threshold=0.5465437)
        true_mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE) // 255
        true_mask_flat = true_mask.reshape(-1)

        acc = compute_global_accuracy(pred_mask_flat, true_mask_flat)

        #enc = do_length_encode(mask)

        #rle_mask.append(' '.join(map(str, enc)))
        acc_list.append(acc)

    df = pd.DataFrame({ 'id' : id_ , 'acc' : acc_list}).astype(str)
    df.to_csv(csv_file, index=False, columns=['id', 'acc'])





if __name__ == "__main__":

    csv_file = "acc_fold5_adjLR_augment_restnet34_params_e101_tls0.00316_vls0.00572_lr0.001087.csv"

    file = "fold5_adjLR_augment_restnet34_params_e101_tls0.00316_vls0.00572_lr0.001087.pth"

    if platform.system() == "Linux":
        model_file_root = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"
        #test_image_path = '/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/test/'
        image_path = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train"

    elif platform.system() == "Darwin":
        model_file_root = "."
        test_image_path = "."

    model_file = os.path.join(model_file_root, file)

    # load parameter
    #net = unet.UNet(in_channel=1, n_classes=1)
    #net = unetresnet.UNetResNet(encoder_depth=152, num_classes=1)
    net = unetresnet.UNetResNet(encoder_depth=34, num_classes=1)
    #net = unetfrog.SaltNet()

    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(model_file))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model_file))

    calculate_acc(net, image_path, csv_file)

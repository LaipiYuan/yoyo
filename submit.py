import os
import pandas as pd
import platform
import cv2
from PIL import Image
from argparse import ArgumentParser

from torch.utils.data.sampler import *
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

try:
    from data.dataset import TGSSaltDataset
    from model import unet, unetresnet, unetfrog
    from common.run_length_encode import *
except ImportError:
    from .data.dataset import TGSSaltDataset
    from .model import unet, unetresnet, unetfrog
    from .common.run_length_encode import *


def create_parser():
    parser = ArgumentParser(description="Salt Submit'")

    parser.add_argument('-d', '--depth', dest='depth', default=34, type=int,
                        help='depth of ResNet')
    parser.add_argument('-t', '--type', dest='image_type', default='pad', type=str,
                        choices=['pad', 'resize'],
                        help='image type pad or resize')

    parser.add_argument('-s', '--submit', dest='submit', default=False,
                        help='call submit funcion')
    parser.add_argument('-p', '--predict', dest='predict', default=False,
                        help='predict mask')
    parser.add_argument('-c', '--calculate', dest='calculate', default=False,
                        help='calculate iou')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                        help='evaluate iou, Best threshold')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def do_resize(img, resize=128):
    img_resize = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    return img_resize


def hwc_to_chw(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        image = np.transpose(image, axes=[2, 0, 1])
    return image


def pad_image(path, mask=False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT, value=0)

    img = hwc_to_chw(img)

    if mask:
        # Convert mask to 0 and 1 format
        # img = img[:, :, 0:1] // 255
        img = img // 255
        #return torch.from_numpy(img).float()
        return img
    else:
        img = img.astype(np.float32) / 255.0
        #return torch.from_numpy(img).float()
        return img


def predict_mask(net, img, image_type, out_threshold=0.5):

    image = torch.from_numpy(img).unsqueeze(0)

    if torch.cuda.is_available():
        image = Variable(image.cuda())
    else:
        image = Variable(image)

    with torch.no_grad():
        mask_pred = net(image)
        mask_prob = F.sigmoid(mask_pred).squeeze(0)

        if image_type == "resize":
            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(101),
                    transforms.ToTensor()
                ]
            )
            mask = tf(mask_prob.cpu())
            mask = mask.squeeze().cpu().numpy()

        elif image_type =="pad":
            mask = mask_prob.data.cpu().numpy()[:, y_min_pad : 128 - y_max_pad, x_min_pad : 128 - x_max_pad]

        mask_np = mask.squeeze()

    return (mask_np > out_threshold).astype(int)


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def run_calculate_iou(train_image_path, train_mask_path, image_type, out_threshold, model_name):
    id_ = []
    pred_num = []
    mask_num = []
    inter = []
    union = []
    threshold = []

    N = len(list(os.listdir(train_image_path)))
    for index, name in enumerate(os.listdir(train_image_path)):
        if index % 500 == 0:
            print('{}/{}'.format(index, N))

        id_.append(str(name)[:-4])

        if image_type == "pad":
            image = pad_image(os.path.join(train_image_path, name))
        elif image_type == "resize":
            image = cv2.imread(os.path.join(train_image_path, name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            image = do_resize(image, resize=128)

        true_mask = cv2.imread(os.path.join(train_mask_path, name), cv2.IMREAD_GRAYSCALE).astype(np.float32) // 255

        pred_mask = predict_mask(model, image, image_type=args.image_type, out_threshold=out_threshold)

        #
        # calculate iou
        pred_num.append(pred_mask.sum())
        mask_num.append(true_mask.sum())
        inter_ = ((pred_mask == 1) & (true_mask == 1)).sum()
        union_ = ((pred_mask == 1) | (true_mask == 1)).sum()
        inter.append(inter_)
        union.append(union_)
        threshold.append(inter_ / (union_ + 1e-12))

    data = pd.DataFrame(data={"id": id_,
                              "pred": pred_num,
                              "mask": mask_num,
                              "inter": inter,
                              "union": union,
                              "threshold": threshold})

    data.to_csv(model_name[:-4] + "_iou.csv", index=False)

    print("Calculate IoU finished")


def run_submit_single(model, model_name, img_root, image_type, out_threshold):

    id_ = []
    rle_mask = []

    N = len(list(os.listdir(img_root)))
    for index, name in enumerate(os.listdir(img_root)):
        if index % 1000 == 0:
            print('{}/{}'.format(index, N))

        id_.append(str(name)[:-4])

        if image_type == "pad":
            image = pad_image(os.path.join(img_root, name))
        elif image_type == "resize":
            image = cv2.imread(os.path.join(img_root, name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            image = do_resize(image, resize=128)

        pred_mask = predict_mask(model, image, image_type, out_threshold=out_threshold)

        enc = rle_encode(pred_mask * 255)
        # enc = do_length_encode(pred_mask)

        rle_mask.append(enc)

    df = pd.DataFrame({'id': id_, 'rle_mask': rle_mask}).astype(str)
    df.to_csv(os.path.join("submit_file", "submit_" + model_name[:-4] + '.csv'), index=False, columns=['id', 'rle_mask'])

    print("Submit finished")


def main(out_threshold=0.5):
    if args.submit:
        print("####################    Image ({})     ####################".format(args.image_type))
        run_submit_single(model, model_params_name, test_img_root, image_type=args.image_type, out_threshold=out_threshold)

    if args.predict:
        input_file = os.path.join(train_img_root, "images", image_file)
        if args.image_type == "pad":
            img = pad_image(input_file)
        elif args.image_type == "resize":
            image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            img = do_resize(image, resize=128)

        mask = predict_mask(model, img, image_type=args.image_type, out_threshold=out_threshold)
        result = mask_to_image(mask)
        result.save(image_file[:-4] + "_output.png")
        print("Predict Finished")

    if args.calculate:
        train_image_path = train_img_root + "images/"
        train_mask_path = train_img_root + "masks/"

        run_calculate_iou(train_image_path, train_mask_path, args.image_type, out_threshold, model_params_name)

    if args.evaluate:
        pass


if __name__ == "__main__":

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

    #
    #
    code_root = os.getcwd()

    args = parse_commandline_args()

    image_file = "911efbb175.png"

    if args.image_type == "pad":
        test_img_root = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/images/"
        train_img_root = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train/"
    elif args.image_type == "resize":
        test_img_root = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/test/"
        train_img_root = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train"

    if not os.path.exists("./submit_file"):
        os.mkdir("./submit_file")

    model_params_name = "fold5_pad_restnet34_BCELoss_e80_tacc0.9869_tls0.00137_vacc0.9768_vls0.00307_lr0.000537.pth"
    model_params_path = os.path.join(code_root, "model_params", model_params_name)


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

    print("**********   Load Model Finished   **********")

    main(out_threshold=0.489548)




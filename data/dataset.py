import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import platform

from torch.utils.data.dataset import Dataset


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

    if mask:
        img = img // 255
        return img.astype(np.float32)
    else:
        img = img / 255.0
        return img.astype(np.float32)


class TGSSaltDataset(Dataset):

    def __init__(self, root_path, file_list, depth_list=None, type="pad", mode="train", transform=None):
        self.root_path = root_path
        self.file_list = file_list
        self.type = type
        self.mode = mode
        self.transform = transform

        # self.ids    = []
        # self.images = []
        # self.masks  = []
        """
        if self.mode in ['train', 'valid']:
            for i in range(len(file_list)):
                file_id = self.file_list[i]
                image_folder = os.path.join(self.root_path, "images")
                image_path = os.path.join(image_folder, file_id + ".png")

                mask_folder = os.path.join(self.root_path, "masks")
                mask_path = os.path.join(mask_folder, file_id + ".png")

                image = np.array(imageio.imread(image_path), dtype=np.uint8)
                mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
                self.images.append(image)
                self.masks.append(mask)
                self.ids.append(file_id)

        elif self.mode in ['test']:
            for i in range(len(file_list)):
                file_id = self.file_list[i]
                image_folder = os.path.join(self.root_path)
                image_path = os.path.join(image_folder, file_id + ".png")

                image = np.array(imageio.imread(image_path), dtype=np.uint8)

                self.images.append(image)
                self.masks.append([])
                self.ids.append(file_id)
        """

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        self.ids = self.file_list[index]

        if self.mode in ['train', 'valid']:
            image_folder = os.path.join(self.root_path, "images")
            image_path = os.path.join(image_folder, self.ids + ".png")

            mask_folder = os.path.join(self.root_path, "masks")
            mask_path = os.path.join(mask_folder, self.ids + ".png")

            if self.type == "pad":
                image = load_image(image_path)
                mask = load_image(mask_path, mask=True)
            elif self.type == "resize":
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

            if self.transform:
                image, mask = self.transform(image, mask)

        elif self.mode in ['test']:
            image_folder = os.path.join(self.root_path)
            image_path = os.path.join(image_folder, self.ids + ".png")

            if self.type == "pad":
                image = load_image(image_path)
            elif self.type == "resize":
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255   # (101, 101)

            mask = []

        image = hwc_to_chw(image)
        mask = hwc_to_chw(mask)

        return image, mask

    def __len__(self):
        return len(self.file_list)


def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')


if __name__ == "__main__":
    mode = "train"

    img_root_path = "."
    code_root_path = "."

    if platform.system() == "Linux":
        img_root_path = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train"
        code_root_path = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"

    elif platform.system() == "Darwin":
        img_root_path = "/Users/liapck/kaggle/TGS/dataset/"
        code_root_path = "/Users/liapck/DeepLearning/Unet/Unet"

    #
    if mode == "train":
        train_df = pd.read_csv(os.path.join(code_root_path, "data", "train_fold_0.csv"))
        file_list = list(train_df['id'].values)
        depth_list = list(train_df['z'].values)

        data_path = os.path.join(img_root_path, "train")

    elif mode == "test":

        train_df = pd.read_csv(os.path.join(code_root_path, "data", "test_depth.csv"))
        file_list = list(train_df['id'].values)
        depth_list = list(train_df['z'].values)

        data_path = os.path.join(img_root_path, "images")

    dataset = TGSSaltDataset(data_path, file_list, depth_list, mode=mode)

    for i in range(2):
        image, mask = dataset[np.random.randint(0, len(dataset))]  # image: (101, 101, 3), mask: (101, 101, 1)
        print(dataset.ids)
        if platform.system() == "Darwin" and mode == "train":
            plot2x2Array(image, mask)
            plt.show()


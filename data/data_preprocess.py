import os
import glob
import cv2
import platform

DST_DIR = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_96"

resize = (96, 96)


def create_image_lists(root_path, mode):
    # key: class name, value: {train images, validation images}
    num_image = 0
    extensions = ["png"]

    if mode == "train":
        result = {}
        DATA_PATH = os.path.join(root_path, "train")

        sub_dirs = [x[0] for x in os.walk(DATA_PATH)]
        is_root_dir = True

        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue

            file_list = []
            dir_name = os.path.basename(sub_dir)
            for extension in extensions:
                file_glob = os.path.join(DATA_PATH, dir_name, '*.' + extension)
                file_list.extend(glob.glob(file_glob))  # glob.glob(pathname), 返回所有匹配的文件路徑列表。
            if not file_list: continue

            images = []
            for file_name in file_list:
                num_image += 1
                base_name = os.path.basename(file_name)
                images.append(base_name)

            result[dir_name] = images

    elif mode == "test":
        result = []
        DATA_PATH = os.path.join(root_path, "images")

        file_list = []
        for extension in extensions:
            file_glob = os.path.join(DATA_PATH, '*.' + extension)
            file_list.extend(glob.glob(file_glob))

        images = []
        for file_name in file_list:
            num_image += 1
            base_name = os.path.basename(file_name)
            images.append(base_name)

        result = images

    return result, num_image


def get_image_path(root_path, image_dict, index, mode, dir_name=None):
    if mode == "train":
        DATA_PATH = os.path.join(root_path, "train")

        image_list = image_dict[dir_name]
        image_name = image_list[index]
        path = os.path.join(DATA_PATH, dir_name, image_name)

    elif mode == "test":
        DATA_PATH = os.path.join(root_path, "images")

        image_name = image_dict[index]
        path = os.path.join(DATA_PATH, image_name)

    return path


def resize_img(root_path, image_dict, mode):
    if mode == "train":
        dir_name_list = list(image_dict.keys())
        for dir_index, dir_name in enumerate(dir_name_list):

            for i in range(len(image_dict[dir_name])):
                image_path = get_image_path(root_path, image_dict, i, mode, dir_name)
                img = cv2.imread(image_path)

                if img is not None:
                    img_resize = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)

                base_name = os.path.basename(image_path)
                save_path = os.path.join(DST_DIR, mode, dir_name)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                cv2.imwrite(os.path.join(save_path, base_name), img_resize, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            print("File \"" + dir_name + "\" is finished!----------- ", len(image_dict[dir_name]))
        print("\n***** Finished ALL *****\n")

    elif mode == "test":
        for i in range(len(image_dict)):
            image_path = get_image_path(root_path, image_dict, i, mode, None)
            img = cv2.imread(image_path)

            if img is not None:
                img_resize = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)

            base_name = os.path.basename(image_path)
            save_path = os.path.join(DST_DIR, mode)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            cv2.imwrite(os.path.join(save_path, base_name), img_resize, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print("\n***** Finished ALL *****\n")


def main(root_path, mode="train"):
    image_lists, image_num = create_image_lists(root_path, mode=mode)
    print(image_lists)
    print("image_num = ", image_num)

    resize_img(root_path, image_lists, mode)


if __name__ == '__main__':
    root_path = "."

    if platform.system() == "Linux":
        root_path = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/"
    elif platform.system() == "Darwin":
        root_path = "/Users/liapck/kaggle/TGS/dataset"

    main(root_path, mode="test")

import os
import pandas as pd
import platform

from sklearn.model_selection import KFold


def generate_csv(root_path):

    train_df = pd.read_csv(os.path.join(root_path, "train.csv"), index_col="id")
    depths_df = pd.read_csv(os.path.join(root_path, "depths.csv"), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df.to_csv("./train_depth.csv")
    test_df.to_csv("./test_depth.csv")


def split_train_valid(root_path, percentage=0.1):
    if not os.path.exists("train_depth.csv"):
        generate_csv(root_path)

    train_df = pd.read_csv("train_depth.csv")

    kf = KFold(n_splits=int(1/percentage), shuffle=True, random_state=True)
    kf.get_n_splits(train_df)

    i = 0
    for train_index, valid_index in kf.split(train_df):
        x_train, x_valid = train_df.iloc[train_index], train_df.iloc[valid_index]

        x_train.to_csv("train_fold_" + str(i) + ".csv", index=False)
        x_valid.to_csv("valid_fold_" + str(i) + ".csv", index=False)
        i += 1


if __name__ == "__main__":

    root_path = "."

    if platform.system() == "Linux":
        root_path = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128"
    elif platform.system() == "Darwin":
        root_path = "/Users/liapck/DeepLearning/Unet/dataset"

    split_train_valid(root_path)
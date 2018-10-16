import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_threshold_iou(file_name, ispad=True):
    if ispad:
        root = "result_pad"
    else:
        root = "result_resize"

    data = pd.read_csv(os.path.join(root, file_name + ".csv"))

    thresholds = data["thresholds"]
    ious = data["ious"]

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]

    threshold_best = thresholds[threshold_best_index]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.savefig(os.path.join("result", file_name + ".png"))


if __name__ == "__main__":

    filename = "threshold_iou_PAD_fold1_e69"

    plot_threshold_iou(filename, ispad=True)
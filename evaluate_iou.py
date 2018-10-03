import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from include import *
    from model import unet, unetresnet, unetfrog
    from data import *
    from data.dataset import TGSSaltDataset
    from common.transform import *
    from common.loss import *
    from trainer import train
except ImportError:
    from .include import *
    from .model import unet, unetresnet, unetfrog
    from .data import *
    from .data.dataset import TGSSaltDataset
    from .common.transform import *
    from .common.loss import *
    from .trainer import train

batch_size = 1
fold = 9


def create_data_loaders(img_root_path, valid_file_list, train_transformation=None):
    valid_dataset = TGSSaltDataset(root_path=img_root_path,
                                   file_list=valid_file_list,
                                   mode="valid",
                                   )

    data_loader = {
        'valid': DataLoader(
            valid_dataset,
            sampler=RandomSampler(valid_dataset),
            batch_size=batch_size,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        ),
    }

    return data_loader


# -----------------------------------------------------------------------------


def do_valid(net, data_loader):
    print("------------ In do_valid -------------")
    val_predictions = []
    val_masks = []

    net = net.eval()

    for image, mask in data_loader['valid']:
        if torch.cuda.is_available():
            image = Variable(image.cuda())  # (BatchSize, 3, H, W)
        else:
            image = Variable(image)

        with torch.no_grad():
            mask_pred = net(image)
            mask_prob = F.sigmoid(mask_pred).cpu().data.numpy()

            val_predictions.append(mask_prob)
            val_masks.append(mask.cpu().data.numpy())

    val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
    val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]
    # print(val_predictions_stacked.shape)

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

    val_predictions_stacked = val_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
    val_masks_stacked = val_masks_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
    # print(val_masks_stacked.shape, val_predictions_stacked.shape)
    iou(val_predictions_stacked, val_masks_stacked)


def iou(val_predictions_stacked, val_masks_stacked):
    metric_by_threshold = []
    for threshold in np.linspace(0, 1, 11):
        val_binary_prediction = (val_predictions_stacked > threshold).astype(int)

        iou_values = []
        for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
            iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
            iou_values.append(iou)
        iou_values = np.array(iou_values)

        accuracies = [
            np.mean(iou_values > iou_threshold)
            for iou_threshold in np.linspace(0.5, 0.95, 10)
        ]

        print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
        metric_by_threshold.append((np.mean(accuracies), threshold))

    best_metric, best_threshold = max(metric_by_threshold)
    print(best_metric, best_threshold)


# -----------------------------------------------------------------------------


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    # print(temp1)
    intersection = temp1[0]
    # print("temp2 = ",temp1[1])
    # print(intersection.shape)
    # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    # print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    # print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def do_valid2(net, data_loader):
    print("------------ In do_valid 2-------------")
    val_predictions = []
    val_masks = []

    net = net.eval()

    for image, mask in data_loader['valid']:
        if torch.cuda.is_available():
            image = Variable(image.cuda())  # (BatchSize, 3, H, W)
        else:
            image = Variable(image)

        mask_pred = net(image)
        mask_prob = F.sigmoid(mask_pred).squeeze(0).squeeze(0).cpu().data.numpy()
        val_predictions.append(mask_prob)
        val_masks.append(mask.squeeze(0).squeeze(0).cpu().data.numpy())

    return np.array(val_predictions), np.array(val_masks)


def evaluate_iou(file_name):
    val_predictions, val_masks = do_valid2(model, data_loader)

    ## Scoring for last model, choose threshold by validation data
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori / (1 - thresholds_ori))

    # ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
    # print(ious)
    ious = np.array([iou_metric_batch(val_masks, val_predictions > threshold) for threshold in thresholds])
    print(ious)

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]

    threshold_best = thresholds[threshold_best_index]

    df = pd.DataFrame(data={"thresholds": thresholds, "ious": ious})
    df.to_csv(file_name[:-4] + "_iou_threshold.csv", index=False)


def plot_threshold_iou(file_name, ispad=True):
    if ispad:
        root = "result_pad"
    else:
        root = "result_resize"

    #data = pd.read_csv(os.path.join(root, file_name[:-4] + "_iou_threshold.csv"))
    data = pd.read_csv(os.path.join(file_name[:-4] + "_iou_threshold.csv"))

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
    plt.savefig(os.path.join("result", file_name[-4] + ".png"))



if __name__ == "__main__":
    code_root_path = os.getcwd()

    model_file_name = "fold1_restnet34_e_params_e69_tls0.00231_vls0.00338_lr0.002499.pth"

    if platform.system() == "Linux":
        # img_root_path = "/home/phymon/liapck/kaggle/TGS_Salt_Identification_128/train"
        img_root_path = "/home/phymon/dataset/kaggle/TGS_Salt_Identification/train"
        model_file_root = "/home/phymon/cloud/julia/kaggle/TGS/Unet/"

    elif platform.system() == "Darwin":
        model_file_root = "."

    valid_df = pd.read_csv(os.path.join(code_root_path, "data", "valid_fold_" + str(fold) + ".csv"))
    valid_file_list = list(valid_df['id'].values)

    data_loader = create_data_loaders(img_root_path, valid_file_list)

    model_file = os.path.join(model_file_root, model_file_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load parameter
    # model = unetfrog.SaltNet()
    model = unetresnet.UNetResNet(encoder_depth=34, num_classes=1)

    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(model_file))
    else:
        model.cpu()
        model.load_state_dict(torch.load(model_file))

    evaluate_iou(model_file_name)


    #
    # Plot          -----------------------------------------------------------

    if not os.path.exists("./result"):
        os.mkdir("./result")

    #plot_threshold_iou(model_file_name, ispad=True)




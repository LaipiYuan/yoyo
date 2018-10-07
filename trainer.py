import os
from datetime import datetime
import numpy as np

from torch.utils.data.sampler import *
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from common.logger import Logger
    from cli import *
except ImportError:
    from .common.logger import Logger
    from .cli import *


logger = Logger('./logs')


def compute_iou2(pred, mask):
    batch_size = mask.shape[0]
    metric = []
    for batch in range(batch_size):
        p, t = pred[batch].data.cpu().numpy() > 0.5, mask[batch].data.cpu().numpy() > 0.5
        
        intersection = np.logical_and(p, t)
        union = np.logical_or(p, t)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)

        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def compute_iou(pred, mask):
    pred = (pred > 0.5).data.cpu().numpy()
    mask = mask.data.cpu().numpy()
    total = len(mask)

    intersection = np.logical_and(pred, mask)
    union = np.logical_or(pred, mask)
    iou = (np.sum(intersection > 0) + 1e-12) / (np.sum(union > 0) + 1e-12)

    return iou


def compute_accuracy(pred, mask):
    pred = (pred > 0.5).data.cpu().numpy().tolist()
    mask = mask.data.cpu().numpy().tolist()

    total = len(mask)
    count = 0.0
    for i in range(total):
        if pred[i] == mask[i]:
            count = count + 1.0
    return float(count) / float(total)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, ini_lr, load=False):
    if load:
        if epoch > 1:
            epoch = epoch + step_in_epoch / total_steps_in_epoch + 80
            lr = ini_lr * 0.99 ** epoch
        else:
            lr = ini_lr
    else:
        if epoch > 15:
            epoch = epoch + step_in_epoch / total_steps_in_epoch
            lr = ini_lr * 0.99 ** epoch
        else:
            lr = ini_lr


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(net, data_loader, data_size, num_epochs, lr, optimizer, criterion, 
            save='acc', fold=0, image_type="pad", loss_type="BCELoss", depth=34, load=False):

    prev_time = datetime.now()
    best_acc = 0.
    best_loss = 10
    _save = False

    for epoch in range(num_epochs):
        # scheduler.step()

        train_loss = 0.
        train_acc = 0.
        train_iou = 0.
        train_iou2 = 0.

        net = net.train()

        #
        # Train     -----------------------------------------------------------

        for i, (image, mask) in enumerate(data_loader['train']):
            adjust_learning_rate(optimizer, epoch, i, len(data_loader['train']), lr, load)

            batch_size = len(image)
            if torch.cuda.is_available():
                image = Variable(image.cuda())
                mask = Variable(mask.cuda())
            else:
                image, mask = Variable(image), Variable(mask)

            # forward propagation
            mask_pred = net(image)
            mask_prob = F.sigmoid(mask_pred)    # logistic
            mask_prob_flat = mask_prob.view(-1)

            true_mask_flat = mask.view(-1)

            if loss_type == "BCELoss":
                loss = criterion(mask_prob_flat, true_mask_flat)    # nn.BCELoss()
            elif loss_type == "FocalLoss":
                loss = criterion(mask_pred, mask, type='sigmoid')  # FocalLoss2d()

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += compute_accuracy(mask_prob_flat, true_mask_flat) * batch_size
            train_iou += compute_iou(mask_prob_flat, true_mask_flat) * batch_size
            train_iou2 += compute_iou2(mask_prob, mask) * batch_size

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)  # math.floor(a/b), a%b
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        _loss = train_loss / data_size['train']
        _acc = train_acc / data_size['train']
        _iou = train_iou / data_size['train']
        _iou2 = train_iou2 / data_size['train']


        #
        # TensorBoard Log     -------------------------------------------------

        # 1. Log scalar values (scalar summary)
        info = { 'loss': _loss, 'accuracy': _acc }
        #info = {'loss': _loss}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            #logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)


        #
        # Validation          -------------------------------------------------

        if data_loader['valid'] is not None:
            valid_loss = 0.
            valid_acc = 0.
            valid_iou = 0.
            valid_iou2 = 0.

            net = net.eval()

            for image, mask in data_loader['valid']:
                batch_size = len(image)
                if torch.cuda.is_available():
                    image = Variable(image.cuda())  # (BatchSize, 3, H, W)
                    mask = Variable(mask.cuda())
                else:
                    image, mask = Variable(image), Variable(mask)

                mask_pred = net(image)
                mask_prob = F.sigmoid(mask_pred)    # logistic

                mask_prob_flat = mask_prob.view(-1)
                true_mask_flat = mask.view(-1)

                if loss_type == "BCELoss":
                    loss = criterion(mask_prob_flat, true_mask_flat)    # nn.BCELoss()
                elif loss_type == "FocalLoss":
                    loss = criterion(mask_pred, mask, type='sigmoid') # FocalLoss2d()

                valid_loss += loss.item()

                """
                _mask = reverse_one_hot(one_hot_it(mask, 2))

                output_image = np.array(mask_pred[0,:,:,:].cpu().detach().numpy())
                output_image = reverse_one_hot(output_image)
                #out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                """
                valid_acc += compute_accuracy(mask_prob_flat, true_mask_flat) * batch_size
                valid_iou += compute_iou(mask_prob_flat, true_mask_flat) * batch_size
                valid_iou2 += compute_iou2(mask_prob, mask) * batch_size

            _val_loss = valid_loss / data_size['valid']
            _val_acc = valid_acc / data_size['valid']
            _val_iou = valid_iou / data_size['valid']
            _val_iou2 = valid_iou2 / data_size['valid']

            """
            epoch_str = ("Epoch %d.  Train Acc: %f, Train Loss: %f, Valid Acc: %f, Valid Loss: %f, lr: %f"
                         % (epoch + 1, _acc, _loss, _val_acc, _val_loss, optimizer.param_groups[0]['lr']))
            """
            epoch_str = ("Epoch %d.  Train Acc: %.4f, Train Loss: %.5f, Train IoU: %.4f  |  Valid Acc: %.4f, Valid Loss: %.5f, Valid IoU: %.4f  |  lr: %.6f"
                         % (epoch + 1, _acc, _loss, _iou, _val_acc, _val_loss, _val_iou, optimizer.param_groups[0]['lr']))

        else:
            epoch_str = ("Epoch %d. Train Loss: %f " % (epoch + 1, _loss))

        #
        # Save      -----------------------------------------------------------
        if save == 'loss':
            if _val_loss < best_loss:
                torch.save(net.state_dict(), os.path.join("model_params",
                           'fold' + str(fold) + '_' + image_type + \
                           '_restnet' + str(depth) + '_'+ loss_type +'_e{}_tacc{:.4f}_tls{:.5f}_vacc{:.4f}_vls{:.5f}_lr{:.6f}.pth'.\
                           format(epoch+1, _acc, _loss, _val_acc, _val_loss, optimizer.param_groups[0]['lr'])))

                _save = True
                best_loss = _val_loss

        elif save == 'acc':
            if _val_acc > best_acc:
                torch.save(net.state_dict(), os.path.join("model_params",
                           'fold' + str(fold) + '_' + image_type + \
                           '_restnet' + str(depth) + '_'+ loss_type+'_e{}_tacc{:.4f}_tls{:.5f}_vacc{:.4f}_vls{:.5f}_lr{:.6f}.pth'.\
                           format(epoch+1, _acc, _loss, _val_acc, _val_loss, optimizer.param_groups[0]['lr'])))

                _save = True
                best_acc = _val_acc

        prev_time = cur_time
        print(epoch_str + ",  " + time_str + ", " + str(int(_save)))
        print("Train IoU: ", _iou2, "\tValid IoU: ", _val_iou2)

        _save = False


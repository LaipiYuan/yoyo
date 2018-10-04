import os
from datetime import datetime

from torch.utils.data.sampler import *
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from common.logger import Logger
except ImportError:
    from .common.logger import Logger

logger = Logger('./logs')


def compute_global_accuracy(pred, label):
    pred = (pred > 0.5).data.cpu().numpy().tolist()
    label = label.data.cpu().numpy().tolist()

    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, ini_lr):
    if epoch > 15:
        epoch = epoch + step_in_epoch / total_steps_in_epoch
        lr = ini_lr * 0.99 ** epoch
    else:
        lr = ini_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(net, data_loader, data_size, num_epochs, lr, optimizer, criterion, save='loss', fold=0, type="pad"):

    prev_time = datetime.now()
    best_acc = 0.
    best_loss = 10
    _save = False

    for epoch in range(num_epochs):
        # scheduler.step()

        train_loss = 0.
        train_acc = 0.
        train_iou = 0.

        net = net.train()

        #
        # Train     -----------------------------------------------------------

        for i, (image, mask) in enumerate(data_loader['train']):
            adjust_learning_rate(optimizer, epoch, i, len(data_loader['train']), lr)

            batch_size = len(image)
            if torch.cuda.is_available():
                image = Variable(image.cuda())  # (BatchSize, 3, H, W)
                mask = Variable(mask.cuda())  # (BatchSize, category), not one-hot-label ?!
            else:
                image, mask = Variable(image), Variable(mask)

            # forward propagation
            mask_pred = net(image)
            mask_prob = F.sigmoid(mask_pred)
            mask_prob_flat = mask_prob.view(-1)

            true_mask_flat = mask.view(-1)

            #loss = criterion(mask_pred, mask, type='sigmoid')
            loss = criterion(mask_prob_flat, true_mask_flat)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            #mask_prob_flat_threshold = (mask_prob_flat > 0.5).astype(int)
            train_acc += compute_global_accuracy(mask_prob_flat, true_mask_flat) * batch_size

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)  # math.floor(a/b), a%b
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        _loss = train_loss / data_size['train']
        _acc = train_acc / data_size['train']


        #
        # TensorBoard         -------------------------------------------------

        # 1. Log scalar values (scalar summary)
        #info = { 'loss': _loss, 'accuracy': _acc }
        info = {'loss': _loss}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            #logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)


        #
        # Validation          -------------------------------------------------

        if data_loader['valid'] is not None:
            valid_loss = 0
            valid_acc = 0
            valid_iou = 0

            net = net.eval()

            for image, mask in data_loader['valid']:
                batch_size = len(image)
                if torch.cuda.is_available():
                    image = Variable(image.cuda())  # (BatchSize, 3, H, W)
                    mask = Variable(mask.cuda())
                else:
                    image, mask = Variable(image), Variable(mask)

                mask_pred = net(image)
                mask_prob = F.sigmoid(mask_pred)
                mask_prob_flat = mask_prob.view(-1)
                true_mask_flat = mask.view(-1)

                # loss = criterion(mask_pred, mask, type='sigmoid')
                loss = criterion(mask_prob_flat, true_mask_flat)

                valid_loss += loss.item()

                """
                _mask = reverse_one_hot(one_hot_it(mask, 2))

                output_image = np.array(mask_pred[0,:,:,:].cpu().detach().numpy())
                output_image = reverse_one_hot(output_image)
                #out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                """
                valid_acc += compute_global_accuracy(mask_prob_flat, true_mask_flat) * batch_size

            _val_loss = valid_loss / data_size['valid']
            _val_acc = valid_acc / data_size['valid']
            """
            epoch_str = ("Epoch %d.  Train Loss: %f, Valid Loss: %f, lr: %f"
                            % (epoch+1, _loss, _val_loss, optimizer.param_groups[0]['lr']))
            """
            epoch_str = ("Epoch %d.  Train Acc: %f, Train Loss: %f, Valid Acc: %f, Valid Lcc: %f, lr: %f"
                         % (epoch + 1, _acc, _loss, _val_acc, _val_loss, optimizer.param_groups[0]['lr']))

        else:
            epoch_str = ("Epoch %d. Train Loss: %f " % (epoch + 1, _loss))

        if save == 'loss':
            if _val_loss < best_loss:
                torch.save(net.state_dict(),
                           'fold' + str(fold) + \
                           '_resize_restnet34_e_params_e{}_tacc{:.4f}_tls{:.5f}_vacc{:.4f}_vls{:.5f}_lr{:.6f}.pth'.\
                           format(epoch+1, _acc, _loss, _val_acc, _val_loss, optimizer.param_groups[0]['lr']))

                _save = True
                best_loss = _val_loss

        prev_time = cur_time
        print(epoch_str + ",  " + time_str + ", " + str(int(_save)))

        _save = False

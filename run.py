import sys
import logging

import torch

#from .main import *
from .cli import parse_dict_args


LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 20,

        # Data
        'dataset': 'cifar10',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,

        # Architecture
        'arch': 'cifar_shakeshake26',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 100.0,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.05,
        'nesterov': True,
    }

    # 4000 labels:
    for data_seed in range(10, 20):
        yield {
            **defaults,
            'title': '4000-label cifar-10',
            'n_labels': 4000,
            'data_seed': data_seed,
            'epochs': 300,
            'lr_rampdown_epochs': 350,
            'ema_decay': 0.99,
        }

    # 1000 labels:
    for data_seed in range(10, 20):
        yield {
            **defaults,
            'title': '1000-label cifar-10',
            'n_labels': 1000,
            'data_seed': data_seed,
            'epochs': 180,
            'lr_rampdown_epochs': 210,
            'ema_decay': 0.97,
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'data-local/labels/cifar10/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    main.args = parse_dict_args(**adapted_args, **kwargs)
    main.main()


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)

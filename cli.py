import re
from optparse import OptionParser
from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser(description="Salt Training'")

    parser.add_argument('-t', '--type', dest='image_type', default='pad', type=str, 
                        choices=['pad', 'resize'],
                        help='image type pad or resize')
    parser.add_argument('-f', '--fold', dest='fold', default=5, type=int,
                        help='number of fold')

    parser.add_argument('-d', '--depth', dest='depth', default=152, type=int,
                        help='depth of ResNet')
    parser.add_argument('-e', '--epochs', dest='epochs', default=300, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate', default=0.005, type=float,
                        help='learning rate')

    parser.add_argument('-v', '--verification', dest='verification', default=False,
                        help='verification model')
    parser.add_argument('-w', '--load', dest='load', default=False,
                        help='load file model')

    parser.add_argument('-l', '--loss', '--consistency-type', dest='consistency_type', default='BCELoss', type=str, 
                        choices=['BCELoss', 'FocalLoss'],
                        help='consistency loss type to use')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))

        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value) for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    return create_parser().parse_args(cmdline_args)


if __name__ == "__main__":

    args = parse_commandline_args()



import re
from optparse import OptionParser
from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser(description="Salt Training'")

    parser.add_argument('-f', '--fold', dest='fold', default=9, type=int,
                        help='number of fold')
    parser.add_argument('-e', '--epochs', dest='epochs', default=150, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=25, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.005, type=float,
                        help='learning rate')
    parser.add_argument('-c', '--load', dest='load', default=False,
                        help='load file model')
    parser.add_argument('-t', '--type', dest='image_type', default='pad', type=str, 
                        help='image type pad or resize')

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
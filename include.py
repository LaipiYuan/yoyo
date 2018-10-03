import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg
print(matplotlib.get_backend())
#print(matplotlib.__version__)


# torch libs
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torchvision import transforms

import torchsummary

from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pydensecrf.densecrf as dcrf


# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from scipy.sparse import coo_matrix

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import platform
from optparse import OptionParser
from argparse import ArgumentParser
import re

# constant #
PI  = np.pi
INF = np.inf
EPS = 1e-12

code_root = os.getcwd()

# Image PAD
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





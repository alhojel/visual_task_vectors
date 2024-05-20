import os.path
from tqdm import trange
import rl_dataloader
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import numpy as np
import os
from contextlib import ExitStack
import torch
import torchvision.transforms as T
from PIL import Image
import pickle
import torch.optim as optim
import torch.nn as nn
import random


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/home/ahojel/datasets/', help='pascal base dir')
    parser.add_argument('--seed', default=15, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--split', default=0 , type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--save_images', default=None, type=int, help='Save images')
    parser.add_argument('--query_support_list_file', default=None, type=str, help='Directory of query support list file')
    parser.add_argument('--train_images', default=100, type=int)
    parser.add_argument('--random_train_images', default=0, type=int)
    parser.add_argument('--eval_iters', default=500, type=int)
    parser.add_argument('--store_latents', default=None, type=str, help='Where to store latents')
    parser.add_argument('--zero_shot', default=0, type=int)
    parser.add_argument('--task', default=None, type=int)
    parser.add_argument('--granularity', default=1, type=int)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--prob', default=0.3, type=float)
    parser.add_argument('--load_model', default=None, type=str, help='Where to load model from')

    return parser


ordered_layer_list = [25,  3, 30, 23, 31, 10, 24, 21, 22, 27, 28, 13, 29,  0, 26,  8, 11,  4, 2,  7,  9, 16,  5, 20,  6, 12,  1, 15, 19, 18, 14, 17]
ordered_head_list = [[25, 2], [30, 7], [28, 3], [27, 8], [21, 2], [3, 9], [24, 3], [26, 10], [24, 15], [16, 8], [30, 0], [3, 11], [30, 3], [28, 9], [29, 15], [0, 0], [10, 2], [26, 0], [26, 7], [29, 4], [22, 14], [31, 13], [25, 7], [29, 2], [0, 10], [23, 13], [29, 7], [30, 5], [30, 11], [10, 12], [23, 8], [31, 1], [12, 5], [30, 8], [30, 14], [11, 11], [27, 3], [28, 13], [30, 12], [22, 0], [23, 5], [25, 10], [24, 4], [13, 6], [27, 0], [0, 14], [22, 9], [24, 7], [24, 10], [23, 14], [4, 12], [29, 12], [31, 5], [26, 14], [27, 9], [6, 4], [30, 2], [16, 13], [31, 11], [31, 12], [22, 11], [11, 13], [31, 15], [5, 2], [31, 0], [27, 4], [8, 2], [21, 15], [27, 14], [5, 14], [25, 1], [22, 13], [11, 5], [12, 2], [3, 4], [3, 7], [28, 6], [25, 15], [2, 8], [13, 1], [5, 13], [29, 8], [24, 5], [31, 10], [7, 12], [6, 2], [28, 5], [27, 1], [9, 15], [31, 8], [17, 14], [28, 0], [8, 9], [26, 6], [5, 3], [31, 7], [15, 3], [31, 6], [4, 15], [21, 8], [10, 0], [31, 3], [25, 8], [23, 10], [3, 1], [13, 9], [31, 4], [0, 4], [28, 11], [0, 8], [30, 10], [3, 6], [29, 10], [7, 7], [23, 3], [31, 2], [30, 4], [0, 9], [23, 15], [4, 5], [25, 13], [13, 11], [9, 9], [8, 3], [18, 11], [20, 9], [0, 5], [27, 13], [22, 8], [1, 10], [28, 15], [3, 2], [1, 9], [21, 1], [5, 8], [31, 14], [4, 9], [5, 15], [28, 14], [28, 7], [1, 13], [11, 8], [0, 1], [29, 0], [13, 0], [20, 2], [0, 2], [19, 14], [2, 0], [20, 11], [13, 8], [29, 6], [5, 6], [20, 3], [25, 4], [10, 14], [3, 15], [24, 14], [20, 8], [15, 5], [18, 9], [25, 12], [28, 1], [20, 1], [15, 1], [27, 7], [19, 11], [21, 14], [5, 4], [18, 3], [17, 5], [21, 6], [28, 2], [22, 10], [16, 1], [25, 14], [28, 10], [14, 15], [13, 3], [9, 13], [0, 3], [13, 12], [22, 12], [13, 15], [2, 10], [26, 8], [31, 9], [3, 13], [9, 14], [28, 12], [29, 9], [16, 5], [4, 6], [1, 5], [7, 13], [1, 3], [6, 12], [27, 6], [2, 5], [3, 10], [1, 12], [23, 7], [25, 5], [27, 10], [10, 6], [23, 9], [2, 13], [6, 0], [27, 5], [23, 12], [23, 0], [1, 1], [9, 1], [22, 7], [2, 4], [3, 5], [18, 15], [2, 6], [25, 9], [3, 8], [22, 2], [13, 2], [22, 6], [7, 0], [14, 8], [15, 11], [4, 1], [6, 9], [14, 5], [19, 6], [25, 0], [13, 10], [10, 15], [19, 5], [10, 1], [16, 2], [20, 15], [2, 14], [7, 1], [21, 7], [4, 3], [2, 12], [29, 3], [10, 7], [16, 7], [13, 7], [30, 6], [16, 6], [10, 9], [27, 2], [7, 14], [28, 8], [16, 9], [9, 3], [15, 15], [8, 4], [16, 12], [24, 13], [28, 4], [23, 4], [2, 7], [18, 6], [3, 3], [7, 9], [8, 13], [24, 8], [22, 3], [26, 9], [3, 12], [15, 0], [22, 4], [14, 7], [21, 11], [30, 13], [0, 12], [24, 12], [13, 13], [29, 14], [27, 12], [22, 5], [11, 1], [17, 0], [21, 10], [30, 15], [13, 14], [19, 3], [4, 11], [2, 3], [10, 3], [26, 12], [11, 3], [19, 10], [17, 15], [5, 7], [22, 1], [29, 5], [3, 0], [8, 1], [15, 12], [7, 2], [2, 2], [11, 0], [16, 10], [14, 12], [11, 6], [27, 15], [23, 1], [9, 8], [25, 11], [26, 2], [19, 1], [3, 14], [7, 8], [24, 11], [24, 0], [17, 8], [21, 0], [25, 3], [18, 13], [12, 13], [2, 9], [29, 1], [0, 15], [1, 7], [23, 11], [27, 11], [4, 4], [26, 1], [17, 12], [9, 10], [6, 1], [4, 13], [5, 10], [12, 9], [19, 4], [5, 0], [0, 13], [9, 12], [14, 2], [6, 7], [19, 0], [20, 14], [25, 6], [18, 5], [29, 13], [24, 1], [7, 5], [6, 11], [12, 11], [18, 2], [16, 11], [29, 11], [15, 7], [16, 4], [23, 6], [14, 0], [11, 15], [20, 10], [16, 15], [10, 4], [20, 12], [12, 0], [23, 2], [17, 7], [12, 3], [14, 6], [30, 1], [17, 4], [20, 5], [21, 9], [19, 12], [4, 8], [9, 2], [20, 6], [11, 7], [14, 13], [9, 6], [12, 12], [15, 13], [12, 15], [20, 4], [11, 14], [21, 12], [13, 4], [7, 4], [24, 6], [19, 13], [17, 6], [17, 3], [12, 14], [13, 5], [17, 9], [7, 6], [12, 10], [19, 15], [18, 4], [0, 6], [24, 2], [4, 10], [4, 0], [8, 12], [14, 1], [30, 9], [14, 14], [4, 2], [15, 9], [11, 2], [26, 3], [12, 8], [10, 13], [5, 5], [26, 15], [19, 2], [8, 0], [4, 14], [12, 7], [21, 3], [11, 12], [18, 0], [11, 4], [6, 5], [10, 10], [15, 6], [14, 9], [9, 5], [9, 7], [15, 10], [10, 5], [8, 6], [16, 0], [8, 7], [4, 7], [18, 10], [14, 4], [12, 6], [11, 10], [16, 3], [15, 8], [22, 15], [20, 7], [6, 15], [17, 13], [8, 8], [8, 14], [19, 8], [19, 7], [8, 10], [17, 2], [10, 8], [14, 3], [1, 2], [6, 10], [6, 6], [8, 11], [12, 1], [10, 11], [20, 0], [18, 12], [15, 14], [7, 15], [21, 5], [2, 1], [14, 11], [15, 2], [14, 10], [1, 6], [6, 13], [21, 13], [17, 10], [24, 9], [8, 15], [19, 9], [9, 11], [7, 11], [18, 8], [21, 4], [5, 11], [12, 4], [15, 4], [2, 15], [7, 3], [6, 14], [7, 10], [11, 9], [18, 14], [9, 0], [18, 7], [9, 4], [8, 5], [17, 1], [20, 13], [1, 11], [26, 11], [16, 14], [18, 1], [5, 1], [0, 7], [6, 3], [2, 11], [26, 13], [17, 11], [26, 5], [5, 9], [26, 4], [1, 8], [6, 8], [5, 12], [1, 0], [0, 11], [1, 15], [1, 4], [1, 14]]

ordered_layer_list_structure = [ 5, 10,  4, 25,  3,  6,  8,  7, 22, 23, 24,  9, 21,  2, 27,  1, 11, 16, 13, 29, 17, 19, 18, 12, 15, 30, 20, 26, 14, 31, 28,  0]
ordered_head_list_structure = [[5, 14], [10, 12], [6, 0], [25, 2], [5, 7], [8, 3], [4, 4], [21, 15], [29, 15], [25, 0], [17, 14], [1, 13], [9, 2], [1, 9], [6, 11], [3, 8], [7, 1], [23, 15], [27, 8], [2, 6], [1, 5], [13, 15], [3, 10], [10, 9], [4, 6], [26, 7], [24, 15], [9, 10], [10, 0], [11, 3], [4, 1], [5, 10], [8, 1], [19, 14], [18, 13], [4, 15], [16, 7], [18, 15], [30, 7], [10, 2], [7, 7], [24, 6], [3, 15], [3, 2], [8, 4], [16, 5], [2, 12], [22, 12], [15, 5], [20, 9], [9, 12], [2, 0], [7, 8], [13, 9], [23, 12], [6, 1], [13, 3], [6, 7], [3, 5], [23, 6], [19, 1], [22, 5], [24, 1], [0, 8], [5, 0], [22, 11], [17, 12], [29, 7], [4, 10], [12, 11], [25, 8], [15, 3], [29, 12], [7, 6], [7, 0], [3, 6], [26, 0], [6, 2], [11, 8], [4, 11], [19, 0], [17, 5], [25, 15], [30, 12], [3, 11], [22, 4], [1, 10], [7, 4], [12, 5], [24, 7], [11, 14], [23, 1], [9, 8], [12, 13], [8, 11], [3, 3], [22, 8], [2, 14], [21, 11], [27, 3], [2, 7], [8, 8], [15, 0], [16, 8], [22, 0], [18, 3], [27, 1], [21, 0], [9, 6], [6, 6], [8, 9], [9, 4], [5, 13], [12, 15], [8, 10], [16, 13], [3, 14], [10, 6], [2, 2], [7, 2], [5, 5], [6, 5], [4, 2], [16, 9], [30, 5], [18, 11], [1, 1], [27, 0], [23, 4], [26, 14], [27, 4], [21, 13], [9, 9], [21, 12], [28, 9], [4, 0], [24, 2], [0, 10], [25, 11], [24, 9], [19, 6], [11, 5], [7, 9], [30, 0], [7, 14], [10, 13], [2, 5], [3, 13], [14, 15], [5, 4], [22, 10], [23, 14], [18, 5], [3, 12], [27, 6], [22, 2], [30, 11], [4, 3], [30, 2], [22, 14], [23, 9], [20, 3], [23, 0], [25, 6], [27, 13], [12, 8], [11, 1], [8, 12], [20, 10], [22, 6], [4, 7], [21, 6], [13, 10], [15, 15], [17, 8], [8, 14], [10, 10], [26, 6], [7, 3], [25, 5], [25, 1], [4, 8], [5, 11], [25, 10], [15, 9], [8, 7], [25, 9], [29, 10], [21, 10], [21, 1], [26, 8], [2, 4], [21, 2], [24, 8], [29, 0], [25, 3], [23, 11], [5, 6], [2, 3], [7, 10], [31, 0], [8, 0], [21, 9], [24, 10], [26, 2], [22, 7], [0, 3], [10, 5], [6, 3], [21, 14], [19, 8], [31, 5], [15, 13], [31, 8], [11, 2], [6, 14], [18, 9], [8, 15], [21, 7], [25, 12], [31, 7], [23, 5], [20, 7], [25, 7], [3, 0], [24, 4], [8, 6], [13, 13], [24, 5], [7, 13], [25, 4], [4, 9], [9, 5], [23, 8], [10, 14], [17, 11], [9, 7], [27, 7], [28, 13], [17, 15], [14, 12], [22, 13], [4, 14], [16, 2], [29, 8], [11, 15], [17, 7], [14, 13], [20, 12], [12, 10], [20, 8], [24, 14], [14, 1], [13, 6], [2, 10], [14, 14], [13, 1], [27, 9], [28, 5], [24, 12], [10, 8], [12, 6], [9, 0], [6, 13], [27, 10], [5, 3], [20, 1], [23, 2], [29, 6], [19, 5], [3, 4], [25, 13], [22, 1], [16, 1], [27, 14], [11, 4], [7, 5], [18, 6], [22, 15], [28, 0], [6, 15], [11, 10], [7, 15], [2, 9], [19, 13], [16, 10], [14, 6], [15, 12], [7, 11], [6, 9], [5, 15], [12, 9], [15, 7], [0, 1], [23, 7], [13, 12], [24, 13], [18, 0], [18, 2], [23, 13], [28, 11], [24, 0], [24, 3], [20, 15], [21, 4], [10, 4], [12, 2], [16, 6], [19, 12], [19, 2], [31, 4], [19, 4], [0, 9], [13, 14], [28, 6], [22, 9], [11, 7], [8, 5], [10, 11], [23, 3], [31, 11], [16, 0], [26, 12], [27, 11], [23, 10], [17, 3], [25, 14], [24, 11], [30, 3], [27, 15], [11, 9], [11, 6], [9, 14], [13, 8], [11, 0], [9, 15], [12, 4], [9, 1], [31, 12], [20, 0], [27, 12], [27, 5], [15, 6], [2, 11], [14, 8], [21, 3], [15, 1], [31, 13], [29, 4], [10, 7], [16, 11], [13, 5], [22, 3], [0, 11], [12, 1], [8, 13], [6, 4], [14, 2], [14, 7], [6, 10], [21, 8], [6, 12], [2, 13], [31, 9], [14, 5], [28, 8], [5, 1], [26, 3], [28, 10], [1, 12], [28, 14], [13, 11], [0, 0], [17, 4], [28, 15], [30, 4], [20, 14], [20, 4], [31, 1], [16, 3], [14, 11], [27, 2], [13, 2], [26, 9], [14, 4], [15, 11], [17, 10], [7, 12], [20, 2], [12, 14], [20, 11], [19, 11], [4, 5], [5, 9], [21, 5], [9, 3], [5, 12], [30, 10], [16, 15], [2, 1], [3, 9], [4, 12], [1, 3], [31, 3], [20, 13], [1, 11], [31, 10], [3, 1], [28, 1], [28, 4], [20, 6], [29, 2], [12, 12], [31, 15], [11, 11], [10, 3], [29, 9], [19, 7], [0, 5], [20, 5], [31, 6], [17, 13], [18, 4], [31, 2], [28, 7], [17, 1], [14, 9], [4, 13], [16, 14], [6, 8], [13, 4], [13, 0], [2, 15], [0, 6], [10, 15], [29, 13], [9, 11], [28, 2], [11, 13], [15, 4], [14, 10], [28, 12], [30, 6], [10, 1], [16, 12], [15, 10], [19, 10], [17, 9], [29, 5], [17, 0], [1, 7], [19, 9], [31, 14], [19, 3], [1, 0], [29, 1], [18, 12], [16, 4], [13, 7], [29, 11], [8, 2], [0, 14], [18, 10], [12, 3], [1, 2], [30, 1], [26, 1], [9, 13], [29, 3], [12, 7], [1, 6], [18, 14], [15, 14], [14, 0], [19, 15], [30, 8], [0, 13], [0, 4], [15, 8], [2, 8], [29, 14], [30, 14], [17, 6], [30, 9], [30, 15], [14, 3], [5, 2], [26, 15], [11, 12], [5, 8], [30, 13], [26, 5], [0, 2], [3, 7], [18, 1], [26, 11], [18, 7], [1, 8], [17, 2], [26, 10], [26, 13], [12, 0], [26, 4], [18, 8], [15, 2], [1, 14], [1, 15], [28, 3], [0, 15], [1, 4], [0, 12], [0, 7]]

ordered_layer_list_combined = [25, 10,  3, 23, 24, 27, 21, 22,  4, 30,  8, 13,  5,  7, 29,  9, 31,  6, 2, 26, 11, 28, 16, 19, 20, 17, 12,  0, 18,  1, 15, 14]
ordered_head_list_combined = [[25, 2], [5, 14], [10, 12], [6, 0], [27, 8], [29, 15], [30, 7], [24, 15], [21, 15], [8, 3], [26, 7], [5, 7], [17, 14], [10, 2], [4, 4], [1, 13], [1, 9], [16, 8], [25, 0], [3, 11], [23, 15], [21, 2], [30, 0], [10, 0], [26, 0], [3, 8], [9, 2], [7, 1], [29, 7], [1, 5], [13, 15], [2, 6], [3, 10], [4, 6], [6, 11], [4, 15], [24, 3], [28, 9], [10, 9], [28, 3], [19, 14], [7, 7], [12, 5], [3, 9], [4, 1], [30, 12], [11, 3], [27, 3], [29, 12], [30, 5], [9, 10], [22, 14], [0, 10], [8, 1], [22, 0], [3, 2], [22, 11], [5, 10], [3, 15], [24, 7], [18, 15], [16, 7], [30, 11], [18, 13], [26, 10], [13, 9], [20, 9], [30, 3], [27, 0], [15, 5], [25, 7], [16, 5], [0, 8], [25, 15], [22, 12], [2, 0], [15, 3], [6, 2], [25, 8], [8, 4], [26, 14], [16, 13], [23, 14], [2, 12], [23, 8], [25, 10], [3, 6], [24, 6], [13, 3], [0, 0], [23, 12], [27, 4], [23, 13], [23, 5], [27, 1], [5, 13], [29, 4], [28, 13], [30, 2], [3, 5], [24, 10], [24, 4], [31, 13], [11, 8], [9, 12], [1, 10], [8, 9], [11, 5], [7, 8], [22, 8], [13, 6], [31, 5], [6, 1], [17, 5], [22, 5], [19, 1], [6, 7], [7, 0], [25, 1], [29, 2], [31, 0], [5, 0], [23, 6], [24, 1], [18, 11], [31, 1], [17, 12], [9, 9], [18, 3], [27, 9], [4, 11], [12, 11], [26, 6], [22, 4], [2, 14], [22, 9], [19, 0], [4, 10], [3, 3], [11, 11], [22, 13], [2, 7], [7, 6], [29, 10], [21, 11], [31, 8], [27, 13], [15, 0], [24, 5], [23, 1], [10, 6], [9, 8], [29, 8], [12, 13], [27, 14], [31, 7], [30, 8], [5, 4], [1, 1], [20, 3], [13, 1], [22, 10], [14, 15], [3, 4], [7, 4], [30, 14], [3, 13], [11, 14], [2, 5], [21, 1], [31, 11], [16, 9], [21, 0], [28, 5], [4, 12], [27, 6], [6, 4], [2, 2], [23, 4], [19, 6], [7, 2], [31, 12], [12, 2], [3, 14], [29, 0], [21, 6], [5, 3], [23, 9], [28, 0], [7, 14], [8, 11], [22, 2], [5, 6], [7, 9], [8, 8], [9, 6], [23, 0], [0, 14], [28, 6], [4, 3], [4, 9], [12, 15], [3, 12], [26, 8], [25, 5], [25, 11], [22, 6], [21, 14], [18, 9], [13, 10], [25, 12], [5, 5], [6, 6], [25, 13], [0, 3], [25, 9], [4, 2], [28, 11], [25, 4], [6, 5], [15, 15], [8, 10], [2, 4], [10, 14], [31, 15], [31, 4], [11, 1], [21, 12], [9, 15], [18, 5], [9, 4], [27, 7], [4, 0], [24, 14], [20, 8], [24, 2], [22, 7], [11, 13], [23, 10], [10, 13], [0, 9], [24, 8], [7, 13], [5, 15], [25, 6], [21, 13], [23, 3], [21, 10], [29, 6], [17, 8], [7, 12], [20, 10], [21, 8], [0, 1], [8, 2], [21, 7], [2, 3], [20, 1], [24, 9], [31, 10], [2, 10], [16, 1], [8, 12], [25, 3], [5, 2], [26, 2], [12, 8], [23, 11], [4, 8], [16, 2], [27, 10], [30, 4], [13, 13], [3, 0], [31, 3], [4, 7], [30, 10], [10, 10], [3, 1], [15, 9], [13, 12], [13, 11], [31, 6], [21, 9], [8, 14], [2, 8], [17, 15], [8, 7], [3, 7], [19, 5], [4, 5], [14, 12], [23, 7], [13, 8], [6, 9], [28, 15], [8, 0], [5, 11], [7, 3], [15, 13], [28, 14], [31, 2], [24, 12], [18, 6], [25, 14], [11, 2], [10, 5], [15, 1], [22, 1], [20, 15], [19, 8], [0, 4], [11, 15], [9, 14], [24, 13], [7, 10], [0, 5], [20, 2], [17, 7], [20, 12], [16, 6], [15, 12], [16, 10], [20, 11], [8, 6], [20, 7], [14, 13], [28, 10], [9, 5], [8, 15], [2, 9], [9, 7], [6, 14], [12, 10], [6, 3], [31, 9], [4, 14], [27, 5], [14, 1], [9, 1], [7, 5], [28, 7], [12, 9], [19, 11], [6, 12], [23, 2], [14, 14], [1, 12], [13, 0], [14, 8], [31, 14], [2, 13], [24, 0], [13, 14], [15, 7], [28, 1], [14, 6], [10, 7], [19, 13], [14, 5], [17, 11], [18, 2], [26, 12], [11, 4], [12, 6], [19, 4], [1, 3], [10, 8], [27, 12], [13, 2], [10, 4], [8, 13], [22, 3], [5, 8], [28, 8], [6, 13], [29, 9], [11, 0], [27, 15], [15, 11], [22, 15], [11, 6], [6, 15], [11, 10], [24, 11], [14, 7], [28, 2], [27, 11], [19, 12], [27, 2], [9, 0], [7, 15], [18, 0], [28, 12], [0, 2], [7, 11], [26, 9], [9, 3], [19, 2], [11, 7], [9, 13], [14, 2], [16, 11], [17, 3], [28, 4], [21, 4], [10, 15], [10, 1], [16, 0], [20, 14], [30, 6], [10, 3], [13, 5], [10, 11], [17, 4], [16, 12], [21, 3], [13, 7], [15, 6], [29, 3], [8, 5], [20, 4], [26, 3], [20, 0], [16, 15], [17, 0], [12, 4], [19, 10], [11, 9], [19, 3], [29, 5], [12, 1], [12, 14], [4, 13], [6, 10], [29, 14], [2, 11], [16, 3], [1, 7], [20, 6], [29, 1], [14, 4], [29, 13], [20, 5], [12, 12], [30, 15], [30, 13], [14, 11], [26, 1], [5, 1], [17, 10], [18, 4], [0, 11], [21, 5], [13, 4], [29, 11], [16, 4], [2, 1], [0, 13], [0, 6], [14, 9], [12, 3], [19, 7], [17, 9], [30, 1], [17, 13], [14, 0], [5, 9], [20, 13], [1, 11], [15, 10], [5, 12], [0, 12], [19, 15], [17, 1], [17, 6], [12, 7], [14, 10], [9, 11], [2, 15], [18, 10], [30, 9], [16, 14], [18, 12], [15, 4], [19, 9], [12, 0], [0, 15], [1, 2], [26, 15], [11, 12], [15, 8], [6, 8], [15, 14], [1, 6], [14, 3], [18, 14], [1, 0], [17, 2], [18, 7], [26, 11], [18, 1], [26, 5], [18, 8], [15, 2], [26, 13], [1, 8], [26, 4], [1, 15], [1, 14], [1, 4], [0, 7]]


random_layer_list = random.sample(ordered_layer_list_combined, len(ordered_layer_list_combined))
random_head_list = random.sample(ordered_head_list_combined, len(ordered_head_list_combined))


class JointModel(nn.Module):
    def __init__(self, args, prompting_model, num_variables, train_ds, eval_ds, task_tensor, load_model=None):
        super().__init__()
        self.prompting_model = prompting_model
        self.num_variables = num_variables
        self.bernoullis = [[25,2,2], [25,2,1], [30,7,2], [27,8,2], [10,2,1], [10,12,1], [8,3,1], [8,9,1], [23,15,0], [5,14,1], [6,0,1], [26,7,2], [25,0,2], [24, 15, 2]]

        self.eps = 1e-6
        self.batch_size = 5*args.train_images if args.task is None else args.train_images

        self.train_ds = train_ds
        self.eval_ds = eval_ds

        self.task_tensor = task_tensor

        self.areas_to_check = [None]

        """ if load_model is not None:
            self.bernoullis = pickle.load(open(load_model, 'rb'))
            self.bernoullis = [torch.tensor(bernoulli, requires_grad=True) for bernoulli in self.bernoullis] """

    def get_good_init(self, args, canvases, sample_count, all_areas, prob):

        initializations = []
        for sample in range(sample_count):

            random.shuffle(all_areas)
            rand_select = [area for area in all_areas if random.random() < prob]
            current = rand_select + self.bernoullis
            current = [list(x) for x in set(tuple(x) for x in current)]

            initializations.append(current)

        loss_list = []

        for init in trange(len(initializations)):
            init = initializations[init]
            areas_to_patch = init

            indices = self.construct_indices(areas_to_patch)
            for i in range(0,self.batch_size):

                canvas = canvases[i]

                if args.task is not None:
                    current_injection = self.task_tensor
                else:
                    current_injection = self.task_tensor[i%len(self.task_tensor)]
            
                with torch.no_grad():        
                    if args.zero_shot:
                        indices_premask = []
                        indices_postmask = []

                        drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                        
                        indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                        #indices_premask = list(range(0,148))#
                        indices_premask = [0]+list(range(99,148))

                        if indices is not None:
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result)

                loss_list.append(loss)

        loss_list = torch.tensor(loss_list)
        loss_list_copy = torch.tensor(loss_list)

        if args.task is None:
            for index in range(len(self.task_tensor)):
                current = loss_list[index::len(self.task_tensor)]
                loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)
        else:
            loss_list = (loss_list-loss_list.mean())/(loss_list.std() + self.eps)

        reshaped_tensor = loss_list.reshape(len(initializations), self.batch_size)
        averages = reshaped_tensor.mean(axis=1)
        best_index = torch.argmin(averages)

        best_loss = loss_list_copy.reshape(len(initializations), self.batch_size).mean(axis=1)[best_index]     

        self.bernoullis = initializations[best_index]
        return best_loss, initializations[best_index]
        

    def loss_mse(self, target, ours):
        ours = (torch.permute(ours / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]
        target = (torch.permute(target.to(ours.device) / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]

        target = target[:, 113:, 113:]
        ours = ours[:, 113:, 113:]
        mse = torch.mean((target - ours) ** 2)
        return mse.item()

    def loss_iou(self, original_image, generated_result):
        fg_color=WHITE
        bg_color=BLACK

        original_image = round_image(original_image, [WHITE, BLACK])
        generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

        target = original_image[113:, 113:].to(original_image)
        ours = generated_result[113:, 113:].to(original_image)

        fg_color = torch.tensor(fg_color, dtype=torch.float32, device=target.device)
        seg_orig = ((target - fg_color[None, None, :]) == 0).all(dim=2)
        seg_our = ((ours - fg_color[None, None, :]) == 0).all(dim=2)
        iou = torch.sum(seg_orig & seg_our).float() / torch.sum(seg_orig | seg_our).float()

        return iou
    
    def construct_indices(self, sampled_patches):
        
        """ if args.granularity==0:
            sampled_patches = sampled_patches.reshape(32,16)
        
            indices = sampled_patches.nonzero()
            expanded_indices = []

            for element in indices:
                layer = int(element[0].item())
                head = int(element[1].item())
                if element[0] < 24:
                    for a in range(0, 50 if args.zero_shot else 148):
                        expanded_indices.append([layer, head, a])
                else:
                    for a in range(0, 99 if args.zero_shot else 197):
                        expanded_indices.append([layer, head, a])
            indices = expanded_indices

            return indices
            """
        if sampled_patches==[]:
            return None
        
        if args.granularity==1:
        
            indices = sampled_patches
            expanded_indices = []

            for element in indices:
                
                head = element[0]
                layer = element[1]
                quadrant = element[2]
                
                if quadrant == 0:
                    expanded_indices.append([head, layer, 0])
                elif quadrant == 1:
                    if head<24:
                        for a in range(1, 50):
                            expanded_indices.append([head,layer, a])
                    else:
                        for a in q1:
                            expanded_indices.append([head, layer, a+1])
                elif quadrant == 2:
                    for a in q2:
                        expanded_indices.append([head, layer, a+1])
            indices = expanded_indices

            return indices
        
            """  if args.granularity==2:
        
            indices = sampled_patches.nonzero()
            expanded_indices = []

            for element in indices:
                index = int(element.item())
                total_elements_per_layer_first_24_heads = 16 * 50  # 16 layers, 2 elements per layer
                total_elements_per_layer_next_8_heads = 16 * 99  # 16 layers, 3 elements per layer
                total_elements_first_24_heads = 24 * total_elements_per_layer_first_24_heads
                
                if index < total_elements_first_24_heads:
                    head = index // total_elements_per_layer_first_24_heads
                    layer = (index % total_elements_per_layer_first_24_heads) // 50
                    quadrant = index % 50
                else:
                    adjusted_index = index - total_elements_first_24_heads
                    head = 24 + (adjusted_index // total_elements_per_layer_next_8_heads)
                    layer = (adjusted_index % total_elements_per_layer_next_8_heads) // 99
                    quadrant = adjusted_index % 99
                
                expanded_indices.append([head, layer, quadrant])
            indices = expanded_indices


            return indices """
    
    def forward(self, args, canvases, iter):

        #self.areas_to_check = [list(x) for x in set(tuple(x) for x in self.areas_to_check if x is not None)]
        self.areas_to_check = [x for x in self.areas_to_check if x not in self.bernoullis]
        self.areas_to_check.append(None)  # Adding None back to the list as required

        loss_list = []

        for addition in trange(len(self.areas_to_check)):
            addition = self.areas_to_check[addition]
            save = False
            areas_to_patch = self.bernoullis

            if addition is not None:  
                areas_to_patch = areas_to_patch + [addition]
            else:
                save=True

            indices = self.construct_indices(areas_to_patch)
            for i in range(0,self.batch_size):

                canvas = canvases[i%len(canvases)]

                if args.task is not None:
                    current_injection = self.task_tensor
                else:
                    current_injection = self.task_tensor[i%len(self.task_tensor)]
            
                with torch.no_grad():        
                    if args.zero_shot:
                        indices_premask = []
                        indices_postmask = []

                        drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                        
                        indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                        #indices_premask = list(range(0,148))#
                        indices_premask = [0]+list(range(99,148))

                        if indices is not None:
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result)

                loss_list.append(loss)

                if save and i == 0:
                    image = generated_result.detach().cpu().numpy()
                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')  # Turn off axis numbers and ticks
                    image_save_path = os.path.join(args.output_dir, f'core_{iter}_{len(self.bernoullis)}.png')
                    plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)   # Save the image without padding and no axis
                    plt.close()  # Close the plot to prevent it from displaying in the notebook or script output

        removal_holder = [ele for ele in self.bernoullis if ele[0]==self.areas_to_check[0][0]]
        
        for removal in trange(len(removal_holder)):
            removal = removal_holder[removal]
            save = False
            areas_to_patch = [area for area in self.bernoullis if area != removal]
                
            indices = self.construct_indices(areas_to_patch)
            for i in range(0,self.batch_size):

                canvas = canvases[i%len(canvases)]

                if args.task is not None:
                    current_injection = self.task_tensor
                else:
                    current_injection = self.task_tensor[i%len(self.task_tensor)]
            
                with torch.no_grad():        
                    if args.zero_shot:
                        indices_premask = []
                        indices_postmask = []

                        drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                        
                        indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                        #indices_premask = list(range(0,148))#
                        indices_premask = [0]+list(range(99,148))

                        if indices is not None:
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=current_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                        else: 
                            original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)
                    else:
                        original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=current_injection)
                    
                
                if args.task == 0:
                    loss = -1*self.loss_iou(generated_result, original_image)
                else:
                    if args.task is None:
                        if i%len(self.task_tensor) == 0:
                            loss = -1*self.loss_iou(generated_result, original_image)
                        else:
                            loss = self.loss_mse(original_image, generated_result)
                    else:
                        loss = self.loss_mse(original_image, generated_result)

                loss_list.append(loss)

        loss_list = torch.tensor(loss_list)
        loss_list_copy = torch.tensor(loss_list)

        if args.task is None:
            for index in range(len(self.task_tensor)):
                current = loss_list[index::len(self.task_tensor)]
                loss_list[index::len(self.task_tensor)] = (current-current.mean())/(current.std() + self.eps)
        else:
            loss_list = (loss_list-loss_list.mean())/(loss_list.std() + self.eps)

        reshaped_tensor = loss_list.reshape(len(self.areas_to_check)+len(removal_holder), self.batch_size)
        averages = reshaped_tensor.mean(axis=1)
        
        best_index = torch.argsort(averages, descending=False)[0]

        best_loss = loss_list_copy.reshape(len(self.areas_to_check)+len(removal_holder), self.batch_size).mean(axis=1)[best_index]
        
        if best_index < len(self.areas_to_check):
            best_move = [self.areas_to_check[best_index]]
            print("Added:", best_move)
        else:
            best_move = [removal_holder[best_index-len(self.areas_to_check)]]
            print("Removed:", best_move)

        if best_move[0] is not None:
        
            if best_index < len(self.areas_to_check):
                self.bernoullis = self.bernoullis + best_move
            else:
                self.bernoullis.pop(self.bernoullis.index(best_move[0]))

        return best_loss, generated_result, best_move

    def train(self, args, num_itr):

        canvases = []
        for idx in range(len(self.train_ds)):
            if args.task is not None:
                canvas = self.train_ds[idx]['grid']
                canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

                canvases.append(canvas)
            else:
                canvas = self.train_ds[idx]['grid']
                for element in canvas:
                    element = (element - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
                    canvases.append(element)

        for k in [15,16]: #[]: #[args.k]: 17,8,10,20,12,14 18,19
        
            for prob in [0.2,0.05,0.1,0.3,0.4,0.5,0.6]: #[args.prob]:#
                
                start_layers = ordered_layer_list_structure[:k]
                start_quadrants = []
                
                for layer in start_layers:
                    if layer<=23:
                        start_quadrants += [[layer, head, quadrant] for head in range(0,16) for quadrant in [0,1]]
                    if layer>=24:
                        start_quadrants += [[layer, head, quadrant] for head in range(0,16) for quadrant in [0,1,2]]
                
                init_loss, _ = self.get_good_init(args, canvases, 100, start_quadrants, prob)

                eval_iou, eval_indices = self.eval(args, str(k)+"-"+str(prob))

                print("Layer:",layer,"Eval:",eval_iou)

                with open(os.path.join(args.output_dir,'log.txt'), 'a') as log:
                    current_metric = {}
                    current_metric["k"] = k
                    current_metric["prob"] = prob
                    current_metric["eval_loss"] = eval_iou
                    current_metric["all_areas"] = self.bernoullis

                    log.write(str(current_metric) + '\n')
                
    def eval(self, args, iter):

        if args.task is None:
            curr_injection = self.task_tensor[0]
        else:
            curr_injection = self.task_tensor
        
        indices = self.construct_indices(self.bernoullis)
        
        loss_holder = []
        for idx in trange(len(self.eval_ds)):

            canvas = self.eval_ds[idx]['grid']
            canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

            with torch.no_grad():        
                if args.zero_shot:
                    indices_premask = []
                    indices_postmask = []

                    drop_indices = [1+a for a in calculate_quadrant_indices(14, 14, 1)]+[1+a for a in calculate_quadrant_indices(14, 14, 2)]
                    
                    indices_postmask += [1+a for a in calculate_quadrant_indices(14, 14, 3)] + [1+a for a in calculate_quadrant_indices(14, 14, 4)]
                    #indices_premask = list(range(0,148))#
                    indices_premask = [0]+list(range(99,148))

                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, premask_pass_indices = indices_premask, attention_heads=indices, attention_injection=curr_injection, postmask_pass_indices = indices_postmask, drop_indices = drop_indices)

                    if idx%20==0:
                    
                        image = generated_result.detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(image)
                        plt.axis('off')  # Turn off axis numbers and ticks
                        image_save_dir = os.path.join(args.output_dir, 'rip')
                        if not os.path.exists(image_save_dir):
                            os.makedirs(image_save_dir)
                        image_save_path = os.path.join(image_save_dir, f'train_{iter}_{idx}.png')
                        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        plt.close()  # Close the plot to prevent it from displaying in the notebook or script output


                        image = original_image.detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(image)
                        plt.axis('off')  # Turn off axis numbers and ticks
                        image_save_dir = os.path.join(args.output_dir, 'rip')
                        if not os.path.exists(image_save_dir):
                            os.makedirs(image_save_dir)
                        image_save_path = os.path.join(image_save_dir, f'train_{iter}_{idx}_original.png')
                        plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)  # Save the image without padding and no axis
                        plt.close()  # Close the plot to prevent it from displaying in the notebook or script output
                 
                else:
                    original_image, generated_result, _ = _generate_result_for_canvas(args, self.prompting_model, canvas, attention_heads=indices, attention_injection=self.task_tensor)
                
            if args.task is None:
                #import pdb; breakpoint()
                loss = self.loss_iou(original_image, generated_result).item()
            elif args.task == 0:
                loss = self.loss_iou(original_image, generated_result).item()
            else:
                loss = self.loss_mse(original_image, generated_result)
            loss_holder.append(loss)

        eval_mean_iou = np.mean(loss_holder)

        return eval_mean_iou, len(indices)

def _generate_result_for_canvas(args, model, canvas, premask_pass_indices = None, postmask_pass_indices = None, attention_heads=None, attention_injection=None, drop_indices = None):
    """canvas is already in the right range."""

    ids_shuffle, len_keep = generate_mask_for_evaluation()
    if attention_heads is not None:
        attention_heads = torch.tensor(attention_heads, dtype=torch.int64).to(args.device)
        

    _, im_paste, _, latents = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device, premask_pass_indices = premask_pass_indices, postmask_pass_indices = postmask_pass_indices, attention_heads = attention_heads, attention_injection = attention_injection, record=False, drop_indices = drop_indices)

    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return canvas, im_paste, latents

def evaluate(args):
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = [torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()]), torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.Grayscale(3),
         torchvision.transforms.ToTensor()])]

    ds = rl_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.train_images, type="trn", task=args.task)
    
    eval_ds = rl_dataloader.DatasetPASCAL(args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, query_support_list_file=args.query_support_list_file, iters=args.eval_iters, type="val", task=args.task if not None else 0)
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)

    with open('/home/ahojel/visual_prompting_vid/task_vectors.pkl', 'rb') as f:
        injection_master = pickle.load(f)

    tasks = ["segmentation", "lowlight enhance", "segmentation_neutral", "inpaint single random", "colorization"]
    
    if args.task is not None:
        task = tasks[args.task]
    
        enc_inj = torch.tensor(injection_master["encoder"][task]).to(args.device)
        dec_inj = torch.tensor(injection_master["decoder"][task]).to(args.device)
                
        injection = [enc_inj,dec_inj]
    else:
        injection = []
        for task_element in tasks:
            enc_inj = torch.tensor(injection_master["encoder"][task_element]).to(args.device)
            dec_inj = torch.tensor(injection_master["decoder"][task_element]).to(args.device)
            injection.append([enc_inj,dec_inj])

    
    if args.granularity==0:
        params = 24*16+8*16
    elif args.granularity==1:
        params = 24*16*2+8*16*3
    elif args.granularity==2:
        params = 24*16*50+8*16*99

    rl_model = JointModel(args, model, params, ds, eval_ds, injection, args.load_model)
    rl_model = rl_model.to(args.device)

    
    rl_model.train(args, num_itr=60)

def determine_quartile(z, q1, q2, q3, q4):
    """Determine the quartile group for a given z value."""
    holder = z
    z = z[-1]
    if holder[0]<=23:
        if z == 0:
            return 'q0'
        elif z-1 in q1:
            return 'q1'
        elif z-1 in q2:
            return 'q2'
        else:
            return 'q3'
    else:
        if z == 0:
            return 'q0'
        elif z-1 in q1:
            return 'q1'
        elif z-1 in q2:
            return 'q2'
        elif z-1 in q3:
            return 'q3'
        elif z-1 in q4:
            return 'q4'

def rank_coordinates_fine_grained(coord_value_pairs, q1, q2, q3, q4):
    # Unzip the list of pairs into separate lists
    coordinates, values = zip(*coord_value_pairs)
    
    # Group coordinates by (x, y) and then by quartile
    groups = defaultdict(lambda: defaultdict(list))
    for coord, value in zip(coordinates, values):
        xy_group = tuple(coord[:2])
        quartile = determine_quartile(coord, q1, q2, q3, q4)
        groups[xy_group][quartile].append((coord[2], value))
    
    # Calculate average value for each fine-grained group and sort groups by this average
    fine_grained_averages = {}
    for xy_group, quartiles in groups.items():
        for quartile, members in quartiles.items():
            avg = np.mean([value for _, value in members])
            fine_grained_averages[(xy_group, quartile)] = avg
    
    sorted_fine_grained_groups = sorted(fine_grained_averages.keys(), key=lambda x: fine_grained_averages[x], reverse=True)
    
    # Sort members within each fine-grained group by their z value
    for xy_group, quartiles in groups.items():
        for quartile in quartiles:
            quartiles[quartile] = sorted(quartiles[quartile], key=lambda x: x[0])
    
    # Compile the ranked list based on fine-grained group average and then by z within each group
    ranked_list = []
    for group in sorted_fine_grained_groups:
        xy_group, quartile = group
        for z, _ in groups[xy_group][quartile]:
            ranked_list.append([*xy_group, z])
    
    return ranked_list


def rank_coordinates_zipped(coord_value_pairs):
    # Unzip the list of pairs into separate lists
    coordinates, values = zip(*coord_value_pairs)
    
    # Group coordinates by (x, y)
    groups = defaultdict(list)
    for coord, value in zip(coordinates, values):
        groups[tuple(coord[:2])].append((coord[2], value))
    
    # Calculate average value for each group and sort groups by this average
    group_averages = {group: np.mean([value for _, value in members]) for group, members in groups.items()}
    sorted_groups = sorted(group_averages.keys(), key=lambda x: group_averages[x], reverse=True)
    
    # Sort members within each group by their z value
    for group in groups:
        groups[group] = sorted(groups[group], key=lambda x: x[0])
    
    # Compile the ranked list based on group average and then by z within each group
    ranked_list = []
    for group in sorted_groups:
        for z, _ in groups[group]:
            ranked_list.append([*group, z])
    
    return ranked_list

     

def evaluate_segmentation(original_image, generated_result, args):
    if args.purple:
        original_image = round_image(original_image, [YELLOW, PURPLE])
    else:
        original_image = round_image(original_image, [WHITE, BLACK])

    if args.purple:
        generated_result = round_image(generated_result, [YELLOW, PURPLE], t=args.t)
    else:
        generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

    if args.purple:
        current_metric = calculate_metric(args, original_image, generated_result, fg_color=YELLOW, bg_color=PURPLE)
    else:
        current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)

    return current_metric
    
def evaluate_mse(target, ours):
    ours = (torch.permute(ours / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]
    target = (torch.permute(target.to(ours.device) / 255., (2, 0, 1)) - torch.tensor(imagenet_mean, dtype=torch.float32).to(ours.device)[:, None, None]) / torch.tensor(imagenet_std, dtype=torch.float32).to(ours.device)[:, None, None]

    target = target[:, 113:, 113:]
    ours = ours[:, 113:, 113:]
    mse = torch.mean((target - ours) ** 2)
    return mse.item()

def calculate_quadrant_indices(rows, cols, quadrant):
    """
    Calculate the start and end indices for each quadrant in the flattened tensor.
    """
    row_start, row_end = 0, 7
    col_start, col_end = 0, 7
    
    if quadrant == 2:  # Top Right
        col_start, col_end = 7, 14
    elif quadrant == 3:  # Bottom Left
        row_start, row_end = 7, 14
    elif quadrant == 4:  # Bottom Right
        row_start, row_end = 7, 14
        col_start, col_end = 7, 14

    indices = []
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            index = row * rows + col
            indices.append(index)
    
    return indices


q1 = calculate_quadrant_indices(14, 14, 1)
q2 = calculate_quadrant_indices(14, 14, 2)
q3 = calculate_quadrant_indices(14, 14, 3)
q4 = calculate_quadrant_indices(14, 14, 4)


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)




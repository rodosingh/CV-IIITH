import torch
from torchvision import transforms

import os
import glob
from tqdm import tqdm
from PIL import Image
from scipy.misc import imresize
from test_MATNet import flip

from modules.MATNet import Encoder, Decoder
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch
import argparse

import pydensecrf.densecrf as dcrf
import numpy as np
import sys
import time

import os
from tqdm import tqdm
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

from os import listdir, makedirs
from os.path import isfile, join

def vid2imgs():
    """"""
    pass


inputRes = (473, 473)
use_flip = True

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_transforms = transforms.Compose([to_tensor, normalize])

if __name__ == "__main__":
    output_path = args.output_path
    vid_path = args.vid_path
    tmp_img_folder_path = vid2imgs(vid_path)
    flow_results_path = img2flow(tmp_img_folder_path)
    tmp_out_path = segment(tmp_img_folder_path, flow_results_path)
    imgs2vid(tmp_out_path, output_path, fps=5)
    # del tmp_img_folder_path/contents
    # del tmp_out_path/contents

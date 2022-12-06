"""
This code allows you to obtain classes mapping for trained pmo models.
A mapping json will be stored for further use.

Author: Weiduo Liao
Date: 2022.12.6
"""

import os
import sys
import pickle
import copy
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator, device, devices, set_determ, check_dir
from config import args

from pmo_utils import Pool, Mixer, prototype_similarity, cal_hv_loss, cal_hv

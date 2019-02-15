'''
    Example to read in needle-master images, demos, and environments.

'''
import pytorch_datasets
from tqdm import tqdm
import signal
import sys
import numpy as np
import torch
import torchvision
import multiprocessing
from pdb import set_trace as woah

''' ------------------------------------- '''
nm_dataset = pytorch_datasets.NeedleMaster('/home/molly/workspace/Surgical_Automation/experiments/needle_master_tools/', train_split=None)
woah()

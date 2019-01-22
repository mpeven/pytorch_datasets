'''
MISTIC dataset

Dataset Information
-------------------
Contains videos of surgeons performing a running suture on a phantom.
Annotations include the current maneuver/task (Suturing, Knot Tying, etc.)
More info here: https://projects.lcsr.jhu.edu/hmm/main/index.php/DataSets/MISTIC


Maneuvers
---------
Suture_Throw
Grasp_Pull_Run_Suture
Inter_Maneuver_Segment
Knot_PSM1_1Loop_ACW
Knot_PSM1_1Loop_CW
Knot_PSM1_2Loop_CW
Knot_PSM2_1Loop_CW
Knot_PSM2_1Loop_ACW
Knot_PSM2_2Loop_CW
Knot_PSM2_2Loop_ACW
Undo_Maneuver


User, Trial data
----------------
02 [t052, t147, t192, t236, t302, t319, t489, t534, t585, t668, t711, t777, t808, t917, t918, t977]
03 [t140, t394, t710]
04 [t045, t157, t401, t540, t717, t761]
05 [t477, t521]
06 [t447, t931, t946]
07 [t035, t087, t238, t776]
08 [t048, t586]
09 [t036, t200, t344, t359, t603, t951]
10 [t797, t989]
11 [t110, t451, t704]
12 [t067, t409, t565, t881]
13 [t101, t134, t195, t386, t455, t486, t577, t768, t964]
15 [t283, t377, t438, t497, t627, t705, t844, t850, t899]
17 [t050, t251, t257, t297, t346, t676, t992]
19 [t227, t322, t357, t990]
24 [t042, t047, t120, t348, t612, t690, t841, t933]
30 [t242, t412, t780, t840, t912, t988]
31 [t165, t232, t446, t463, t539, t807, t859]
'''


import os
import glob
import click
from tqdm import tqdm
import subprocess
import scipy.io as sio
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from pytorch_datasets.dataset import DataSet


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MISTIC(DataSet):
    """ Can specify train users with the 'train_users' parameter. Otherwise it uses the default """
    def __init__(self, root, train_split, train_users=None, transforms=None):
        super().__init__(transforms)
        self.root = root
        self.video_frames_location = os.path.join(root, 'video_frames')
        self.train_split = train_split
        self.train_users = self.get_training_split(train_users)

        # Make sure dataset is good to go
        if not self._check_exists():
            raise RuntimeError('Dataset not found at {}'.format(root))
        if not self._check_frames_exists():
            if click.confirm("Do you want convert the MISTIC .avi video files to images? "
                             + "(takes 59GB of space)"):
                self.extract_video_frames()

        # Get maneuver index to name dict
        self.idx_to_name = self.get_index_to_name_dict()

        # Create dataset
        self.dataset = self.get_all_trials()

        # Add users and remove those not in split
        self.dataset = self.add_user_id(self.dataset)
        self.dataset = [x for x in self.dataset if x['user_id'] in self.train_users]

        # Add maneuvers and kinematics
        self.dataset = self.add_maneuver_annotations(self.dataset)
        self.dataset = self.add_kinematics(self.dataset)

    def get_index_to_name_dict(self):
        # Create dictionary of label index to value
        idx_to_name_file = os.path.join(self.root, 'meta_files', 'maneuver_names.txt')
        idx_to_name = pd.read_csv(idx_to_name_file, sep=' ', names=["idx", "name"])
        idx_to_name = dict(zip(idx_to_name.idx, idx_to_name.name))
        idx_to_name[0] = 'No_Maneuver'
        return idx_to_name

    def get_all_trials(self):
        dataset = []
        for vid in sorted(os.listdir(os.path.join(self.root, "video_files"))):
            dataset.append({'trial': vid})
        return dataset

    def add_maneuver_annotations(self, dataset):
        for d in dataset:
            maneuver_file = os.path.join(self.root, 'meta_files', d['trial'], 'maneuvers_framestamp.txt')
            if not os.path.isfile(maneuver_file):
                d['contains_maneuver_annotations'] = False
                continue
            d['contains_maneuver_annotations'] = True
            df = pd.read_csv(maneuver_file, names=["frame", "maneuver_index"], sep=" ")
            d['frames'] = df.frame.values
            d['maneuver_indices'] = df.maneuver_index.values
            d['maneuver_names'] = df.maneuver_index.apply(lambda x: self.idx_to_name[x]).values
        return dataset

    def add_user_id(self, dataset):
        meta_file = sio.loadmat(os.path.join(self.root, 'DataSetMetaInfo.mat'))
        trial_to_user = {a[-1][0]: int(a[-2][0][4:6]) for a in meta_file["TrialData"][0]}
        for d in dataset:
            d['user_id'] = trial_to_user[d['trial']]
        return dataset

    def add_kinematics(self, dataset):
        PSM_USECOLS = [1, 2, 3,
                       4, 5, 6, 7, 8, 9, 10, 11, 12,
                       13, 14, 15,
                       16, 17, 18,
                       25]
        PSM_COL_NAMES = ['pos_x', 'pos_y', 'pos_z',
                         'r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22',
                         'vel_x', 'vel_y', 'vel_z',
                         'omega_x', 'omega_y', 'omega_z',
                         'gripper']
        for d in tqdm(dataset, ncols=115, desc="Reading motion files"):
            motion_file = os.path.join(self.root, 'motion_files', d['trial'], 'API-PSM1.txt')
            if not os.path.isfile(motion_file):
                d['contains_kinematics'] = False
                continue
            d['contains_kinematics'] = True
            df = pd.read_csv(motion_file, sep=" ", usecols=PSM_USECOLS, names=PSM_COL_NAMES)
            d.update({col:np.array(val) for col,val in df.to_dict('list').items()})

        return dataset

    def _check_exists(self):
        return os.path.isdir(self.root) and \
            os.path.isdir(self.root + "/meta_files") and \
            (os.path.isdir(self.root + "/motion_files") or os.path.isdir(self.root + "/video_files"))

    def _check_frames_exists(self):
        return os.path.isdir(self.video_frames_location) and \
            (len(os.listdir(self.video_frames_location)) == 396)

    def extract_video_frames(self):
        # Make folder for extraction
        if not os.path.isdir(self.video_frames_location):
            os.mkdir(self.video_frames_location)

        # Get all videos
        all_vids = sorted(glob.glob("{}/video_files/*/*.avi".format(self.root)))

        # Iterate through all videos and extract frames using ffmpeg
        vid_iterator = tqdm(iterable=all_vids, ncols=100)
        for vid_path in vid_iterator:
            vid_id = "{}_{}".format(
                vid_path.split("/")[-2],
                os.path.basename(vid_path).replace(".avi", "")
            )
            vid_iterator.set_postfix(Extracting=vid_id)

            # Make directory for the individual video frames
            individual_vid_frames_dir = "{}/{}".format(self.video_frames_location, vid_id)
            if not os.path.isdir(individual_vid_frames_dir):
                os.mkdir(individual_vid_frames_dir)

            # Extract frames
            cmd = "ffmpeg -hide_banner -loglevel panic -i {} {}/%05d.jpg".format(
                vid_path, individual_vid_frames_dir
            )
            subprocess.call(cmd, shell=True)

    def get_training_split(self, train_users):
        '''
        Set the train/test/val split between users in:
            [02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 15, 17, 19, 24, 30, 31]
        '''

        # Set the train split users
        if train_users is not None:
            print("MISTIC {} user IDs = {}".format(self.train_split, train_users))
        else:
            # Not defined - set defaults
            if self.train_split == "train":
                train_users = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]
            elif self.train_split == "val":
                train_users = [5, 8, 10]
            elif self.train_split == "test":
                train_users = [19, 24, 30, 31]
            print("MISTIC {} users not specified".format(self.train_split) +
                  " - using default: {}".format(train_users))

        return train_users

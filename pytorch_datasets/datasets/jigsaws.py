'''
JIGSAWS Dataset.

TODO:
    Add in kinematic data
'''

import os
import glob
import click
from tqdm import tqdm
import subprocess
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from pytorch_datasets.dataset import DataSet

class JIGSAWS(DataSet):
    tasks = ['Knot_Tying', 'Suturing', 'Needle_Passing']
    gesture_to_description = {
        "G1" : "reaching_for_needle_with_right_hand",
        "G2" : "positioning_needle",
        "G3" : "pushing_needle_through_tissue",
        "G4" : "transferring_needle_from_left_to_right",
        "G5" : "moving_to_center_with_needle_in_grip",
        "G6" : "pulling_suture_with_left_hand",
        "G7" : "pulling_suture_with_right_hand",
        "G8" : "orienting_needle",
        "G9" : "using_right_hand_to_help_tighten_suture",
        "G10" : "loosening_more_suture",
        "G11" : "dropping_suture_at_end_and_moving_to_end_points",
        "G12" : "reaching_for_needle_with_left_hand",
        "G13" : "making_c_loop_around_right_hand",
        "G14" : "reaching_for_suture_with_right_hand",
        "G15" : "pulling_suture_with_both_hands",
    }

    def __init__(self, root, train_split, train_users=None, transforms=None):
        super().__init__(transforms)
        self.root = root
        self.video_frames_location = os.path.join(root, 'video_frames')
        self.train_split = train_split
        self.train_users = self.get_training_split(train_split, train_users)

        # Make sure dataset is good to go
        if not self._check_exists():
            raise RuntimeError('Dataset not found at {}'.format(root))
        if not self._check_frames_exists():
            if click.confirm("Do you want convert the JIGSAWS .avi video files to images? " +
                                 "(takes 6.1GB of space)"):
                self.extract_video_frames()

        # Create the base info for the dataset (just the trails)
        self.dataset = self.create_dataset()

        # Set the training split
        self.dataset = [x for x in self.dataset if x['user_id'] in self.train_users]

    def create_dataset(self):
        ''' Crawl through the metadata files to create a dataframe of the JIGSAWS dataset '''
        dataset = []
        # Iterate through each task ('Knot_Tying', 'Suturing', 'Needle_Passing')
        for task in self.tasks:
            task_df = pd.read_csv(
                '{0}/{1}/meta_file_{1}.txt'.format(self.root, task),
                sep="\t",
                header=None,
                usecols=[0,3],
                names=["trial", "score"]
            )

            # Iterate through each trial
            for idx, trial in task_df.iterrows():
                task_df = pd.read_csv(
                    '{}/{}/transcriptions/{}.txt'.format(self.root, task, trial["trial"]),
                    sep=" ", index_col=False, names=["frame_start", "frame_stop", "gesture"]
                )

                # Get gestures
                gestures = [{
                    'id': g['gesture'],
                    'frame_start': g['frame_start'],
                    'frame_stop': g['frame_stop']
                } for _,g in task_df.iterrows()]

                dataset.append({
                    'task': task,
                    'trial': trial["trial"],
                    'score': trial["score"],
                    'user_id': re.search("_(\w)\d\d\d", trial["trial"]).group(1),
                    'gestures': gestures,
                })

        return dataset

    def get_training_split(self, train_split, train_users):
        ''' Set the train/test/val split between users in ['B' 'C' 'D' 'E' 'F' 'G' 'H' 'I'] '''

        # Set the train split users
        if train_users != None:
            print("JIGSAWS {} user IDs = {}".format(train_split, train_users))
        else:
            # Not defined - set defaults
            if train_split == "train":
                train_users = ['B','C','D','E','F','G']
            elif train_split == "val":
                train_users = ['H']
            elif train_split == "test":
                train_users = ['I']
            print("JIGSAWS {} users not specified".format(train_split) +
                  " - using default: {}".format(train_users))

        return train_users

    def _check_exists(self):
        ''' Make sure JIGSAWS is downloaded correctly '''
        # Check if dataset location exists
        if not os.path.isdir(self.root):
            return False

        # Make sure necessary files exist
        for task in self.tasks:
            task_folder = "{}/{}".format(self.root, task)
            if not os.path.isdir(task_folder):
                return False

        return True

    def _check_frames_exists(self):
        return os.path.isdir(self.video_frames_location) and \
               len(os.listdir(self.video_frames_location)) == 206

    def extract_video_frames(self):
        # Make folder for extraction
        if not os.path.isdir(self.video_frames_location):
            os.mkdir(self.video_frames_location)

        # Get all videos
        all_vids = sorted(glob.glob("{}/*/video/*".format(self.root)))

        # Iterate through all videos and extract frames using ffmpeg
        vid_iterator = tqdm(iterable=all_vids, ncols=100)
        for vid_path in vid_iterator:
            vid_id = os.path.basename(vid_path).replace(".avi", "")
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

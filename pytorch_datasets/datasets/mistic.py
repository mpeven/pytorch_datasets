'''
MISTIC dataset

Dataset Information
-------------------
Contains videos of surgeons performing a running suture on a phantom.
Annotations include the current phase/task (Suturing, Knot Tying, etc.)
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
import subprocess
import scipy.io as sio
import pandas as pd
import torch
import torchvision


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MISTIC(torch.utils.data.Dataset):
    def __init__(self, root, train_split, train_users=None, per_frame=False):
        self.root = root
        self.video_frames_location = "{}/video_frames".format(root)
        self.train_split = train_split
        self.train_users = train_users
        self.per_frame = per_frame

        # Make sure dataset is good to go
        if not self._check_exists():
            raise RuntimeError('Dataset not found at {}'.format(root))
        if not self._check_frames_exists():
            if click.confirm("Do you want convert the MISTIC .avi video files to images? "
                             + "(takes 59GB of space)"):
                self.extract_video_frames()

        # Collect the dataset
        self.dataset = self.create_dataset()

        # Group into segments
        if not per_frame:
            self.group_by_segments()

        # Set the training split
        self.set_training_split()

        # Filter or add information to dataset
        # self.split_labels_in_two()

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

    def create_dataset(self):
        '''
        Create MISTIC dataset with the following (per-frame) data:
            'frame', 'trial', 'user', 'maneuver_name', 'maneuver_index'
        '''
        # Read all phase-files
        phase_dfs = []
        for phase_file in sorted(glob.glob('{}/meta_files/*/maneuvers_framestamp.txt'.format(self.root))):
            df = pd.read_csv(phase_file, names=["frame", "maneuver_index"], sep=" ")
            df["trial"] = phase_file.split("/")[-2]
            phase_dfs.append(df)
        df = pd.concat(phase_dfs)

        # Add phase-name
        df = df.merge(pd.read_csv(
            '{}/meta_files/maneuver_names.txt'.format(self.root),
            sep=' ', names=["maneuver_index", "maneuver_name"]
        ))

        # Add user name
        x = sio.loadmat("{}/DataSetMetaInfo.mat".format(self.root))
        t2u = [{
            'trial': x["TrialData"][0][a][-1][0],
            'user': int(x["TrialData"][0][a][-2][0][4:6]),
        } for a in range(len(x["TrialData"][0]))]
        df = df.merge(pd.DataFrame(t2u))

        return df

    def set_training_split(self):
        '''
        Set the train/test/val split between users in:
            [02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 15, 17, 19, 24, 30, 31]
        '''

        # Set the train split users
        if self.train_users is not None:
            print("MISTIC {} user IDs = {}".format(self.train_split, self.train_users))
        else:
            # Not defined - set defaults
            if self.train_split == "train":
                self.train_users = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]
            elif self.train_split == "val":
                self.train_users = [5, 8, 10]
            elif self.train_split == "test":
                self.train_users = [19, 24, 30, 31]
            print("MISTIC {} users not specified".format(self.train_split) +
                  " - using default: {}".format(self.train_users))

        # Split the dataset
        self.dataset = self.dataset[self.dataset["user"].isin(self.train_users)].reset_index(drop=True)

    def group_by_segments(self):
        # Make sure it's sorted first
        self.dataset = self.dataset.sort_values(["trial", "frame"])

        # Create unique id to group by
        maneuver_changes = self.dataset["maneuver_index"].diff().ne(0).cumsum()
        frame_jumps = self.dataset["frame"].diff().ne(1).cumsum()
        self.dataset["segment_id"] = (maneuver_changes + frame_jumps).diff().ne(0).cumsum()

        # Pull out all per-segment data
        segments = []
        for _, sub_df in self.dataset.groupby(["segment_id"]):
            # Make sure segment is truly unique
            if len(sub_df["maneuver_name"].unique()) != 1 or \
               len(sub_df["maneuver_index"].unique()) != 1 or \
               len(sub_df["user"].unique()) != 1 or \
               len(sub_df["trial"].unique()) != 1:
                raise RuntimeError("Error combining dataframe")
            segments.append({
                'frame_start': min(sub_df["frame"]),
                'frame_stop': max(sub_df["frame"]),
                'maneuver_name': sub_df["maneuver_name"].unique()[0],
                'maneuver_index': sub_df["maneuver_index"].unique()[0],
                'user': sub_df["user"].unique()[0],
                'trial': sub_df["trial"].unique()[0],
            })

        self.dataset = pd.DataFrame(segments)

    def __len__(self):
        return len(self.dataset)

    def get_image(self, trial, frame):
        # Load image
        hdd_path = "{}/video_frames/{}_right/{:05d}.jpg".format(self.root, trial, frame)
        ssd_path = "/home/mike/Projects/Simulated_Bootstrapping/cached_ims/{}/{:05d}.jpg".format(trial, frame)

        # For MARCC
        # im = Image.open(hdd_path).copy()

        # # Cached image exists
        if os.path.isfile(ssd_path):
            im = pil_loader(ssd_path)
        else:
            im = pil_loader(hdd_path)
            im = transforms.functional.resize(im, (256, 256))
            try:
                im.save(ssd_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(ssd_path)[0])

        # Transform image
        transforms_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop((224, 224)),
            transforms.RandomRotation(30),
            transforms.ColorJitter(.2, .2, .2, .2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transforms_train(im) if self.train_split == "train" else transforms_test(im)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        return {
            'image': self.get_image(row['trial'], row['frame']),
            'user': row['user'],
            'trial': row['trial'],
            'frame': row['frame'],
            'label': row['maneuver_index'],
            'maneuver_name': row['maneuver_name'],
        }

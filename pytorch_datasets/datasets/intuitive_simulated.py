'''
Simulated "Around The World" needle passing videos using the da Vinci Surgical Systemself.
Dataset collected by Anand from 07/23/15 - 07/31/15 during his internship at Intuitive surgical.


Around the World
----------------
Person controls 2 needle drivers.
8 pairs of targets, starts at 3:00 and goes clockwise.
Enter through the flashing target, exit through the solid.


Dataset locations
-----------------
LCSR Server -- lcsr-cirl$/Language Of Surgery Data/ISI-SG-Sim-NP/data/raw/
Titan -- /hdd/Datasets/Intuitive


Dataset info/annotations
------------------------
191 videos
32 different users
Phase annotations (ProgressLog.txt)
Insertion target locations (MetaData.txt/Targets.txt)
Events (SimEvents.txt)
Instrument motion (USM<instrument#>.txt)
Needle motion (EnvMotion.txt )


Important
---------
Timestamps do not line up with actual video time.
For the correct conversion from timestamp in log to frame # in video:
    Get the timestamp from the log file
    Get the closest timestamp from "left_avi_ts.txt"
    Get the corresponding frame number in "left_avi_ts.txt"
    That frame number is the frame in the video
'''


import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


BASE_PATH = '/hdd/Datasets/Intuitive'


class IntuitiveSimulated(Dataset):
    def __init__(self, split):
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'. Get {}".format(split))
        self.split = split
        self.dataset = self.create_dataset()

    def create_dataset(self):

        # Crawl through dataset to get paths
        all_paths = []
        all_videos = sorted(glob.glob('{}/Recorded/*/video_files/*/left.avi'.format(BASE_PATH)))
        for vid in all_videos:
            all_paths.append({
                'video_path': vid,
                'ts_path': vid.replace("left.avi", "left_avi_ts.txt"),
                'phase_path': vid.replace("video_files", "log_files").replace("left.avi", "ProgressLog.txt"),
                'events_path': vid.replace("video_files", "log_files").replace("left.avi", "SimEvents.txt"),
            })

        def get_video_phases(phase_path, ts_path):
            # Get {timestamp, phase} data
            phase_df = pd.read_csv(phase_path, sep=" ", names=["start_time", "event_id", "event_desc"])

            # Get {timestamp, frame} data
            tsp_array = np.loadtxt(ts_path, usecols=1)
            total_frames = len(tsp_array)

            # Convert timestamps to frame numbers
            phase_df["start_frame"] = phase_df["start_time"].apply(lambda ts: np.argmin(np.abs(tsp_array - ts)))
            print(phase_df)

            # Convert {start_frame, phase} dataframe to per-frame labels
            per_frame_labels = []
            print(phase_df)
            print(total_frames)

            return phase_df

        # Crawl through paths to read in data
        dataset = []
        for path_dict in all_paths:
            # Get log events
            phase_df = get_video_phases(path_dict["phase_path"], path_dict["ts_path"])
            exit()

            # for phase in
            dataset.append({
                'video_file': path_dict["video_path"],
                'start_frame': None,
                'end_frame': None,
                'phase': None,
            })

    def __len__(self):
        return len(self.df_dataset)

    def image_transforms(self, numpy_images):
        """ Transformations on a list of images """

        # Get random parameters to apply same transformation to all images in list
        color_jitter = transforms.ColorJitter.get_params(.25, .25, .25, .25)
        rotation_param = transforms.RandomRotation.get_params((-15, 15))
        flip_param = random.random()

        # Apply transformations
        images = []
        for numpy_image in numpy_images:
            i = transforms.functional.to_pil_image(numpy_image)
            i = transforms.functional.resize(i, (224, 224))
            if self.train:
                i = color_jitter(i)
                i = transforms.functional.rotate(i, rotation_param)
                if flip_param < 0.5:
                    i = transforms.functional.hflip(i)
            i = transforms.functional.to_tensor(i)
            i = transforms.functional.normalize(i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            images.append(i)
        return torch.stack(images)

    def pad_or_trim(self, torch_array, bound):
        if torch_array.size(0) <= bound:
            return torch.cat([torch_array, torch.zeros([bound - torch_array.size(0), 3, 224, 224])])

        return torch_array[:bound]

    def __getitem__(self, idx):
        # Max num images
        bound = 500

        # Get path of video
        row = self.df_dataset.iloc[idx].to_dict()
        image_directory = "{}/rgb/{}_{}".format(DATASET_DIR, row['user_id'], row['video_id'])
        op_flow_directory = "{}/op_flow/{}_{}".format(DATASET_DIR, row['user_id'], row['video_id'])

        # Load images
        raw_images = []
        for x in range(row['frame_first'], min(row['frame_last'], row['frame_first'] + bound)):
            im_path = "{}/frame{:08}.jpg".format(image_directory, x)
            try:
                raw_images.append(cv2.imread(im_path)[:, :, ::-1])
            except TypeError:
                # print(row)
                print(im_path)
                print(os.path.isfile(im_path))

        # Transform images
        try:
            transformed_images = self.image_transforms(raw_images)
        except ValueError:
            print("ERROR")
            print(row)
            transformed_images = []

        # Pad or trim to length "bound"
        # print(transformed_images.size())
        padded_images = self.pad_or_trim(transformed_images, bound)

        # Load optical flow
        # raw_optical_flow = []
        # for x in range(row['frame_first'], row['frame_last']+1):
        #     optical_flow_path = "{}/frame{:08}.png".format(op_flow_directory, x)
        #     optical_flow = cv2.imread(optical_flow_path, -1)[:,:,:2]
        #     # Un-quantize and un-bound (bound=1000 -- from jupyter notebook)
        #     optical_flow = (optical_flow * 2000. / 65534.) - 1000.

        # Get label
        label = row['event_id']

        return padded_images, label


if __name__ == '__main__':
    IntuitiveDataset("train")

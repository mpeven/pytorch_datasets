import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch


class WCVP(torch.utils.data.Dataset):
    dset_location = '/hdd/Datasets/WCVP/'

    def __init__(self):
        self.dataset = self.create_dataset()

    def create_dataset(self):
        dataset = []
        for annotation_file in tqdm(glob.glob(self.dset_location + "*/*.txt"),
                                    ncols=100,
                                    desc="Creating WCVP dataset"
                                    ):
            xyz = open(annotation_file, "r").readlines()[3].split()
            dataset.append({
                'image_file': annotation_file.replace('txt', 'jpg'),
                'orientation': np.arctan2(float(xyz[0]), float(xyz[1])) * 180. / np.pi,
                'source': 'WCVP',
            })
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        datum['image'] = Image.open(datum['image_file']).convert('RGB')
        return datum

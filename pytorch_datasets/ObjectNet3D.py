import os
import glob
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
import h5py
import torch
import utils.cache_manager as cache



class ObjectNet3D(torch.utils.data.Dataset):
    dset_location = '/hdd/Datasets/ObjectNet3D/'
    dset_cached_location = dset_location + "cached_dataset.pkl"

    def __init__(self):
        self.dataset = self.create_dataset()

    def create_dataset(self):
        cached_dset = cache.retreive_from_cache(self.dset_cached_location)
        if cached_dset != False:
            return cached_dset

        dataset = []
        for matfile in tqdm(glob.glob(self.dset_location + "Annotations/*")):
            try:
                x = sio.loadmat(matfile)
            except Exception:
                continue

            for obj in x['record']['objects'][0,0][0]:
                # Get elevation (or fine-grained elevation if it exists)
                elevation = obj['viewpoint']['elevation_coarse'][0][0][0][0]
                if 'elevation' in obj['viewpoint'].dtype.names:
                    if len(obj['viewpoint']['elevation'][0][0]) > 0:
                        elevation = obj['viewpoint']['elevation'][0][0][0][0]

                # Get azimuth (or fine-grained azimuth if it exists)
                azimuth = obj['viewpoint']['azimuth_coarse'][0][0][0][0]
                if 'azimuth' in obj['viewpoint'].dtype.names:
                    if len(obj['viewpoint']['azimuth'][0][0]) > 0:
                        azimuth = obj['viewpoint']['azimuth'][0][0][0][0]


                dataset.append({
                    'image_file': self.dset_location + "Images/" + x['record']['filename'][0,0][0],
                    'object_type': obj['class'][0],
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': obj['viewpoint']['distance'][0][0][0][0],
                    'theta': obj['viewpoint']['theta'][0][0][0][0],
                    'bbox': obj['bbox'][0],
                })

        cache.cache(dataset, self.dset_cached_location)
        return dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        datum['image'] = Image.open(datum['image_file']).convert('RGB')
        return datum

import glob
import os
from tqdm import tqdm
import numpy as np
from pytorch_datasets.dataset import DataSet


class WCVP(DataSet):
    def __init__(self, root, transforms=None):
        super().__init__(transforms)
        self.dataset = []
        for annotation_file in glob.glob(os.path.join(root, "*/*.txt")):
            xyz = open(annotation_file, "r").readlines()[3].split()
            self.dataset.append({
                'image_file': annotation_file.replace('txt', 'jpg'),
                'orientation': (int(np.arctan2(float(xyz[0]), float(xyz[1])) * 180. / np.pi) + 360) % 360,
                'source': 'WCVP',
            })

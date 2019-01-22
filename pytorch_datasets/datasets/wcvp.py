import glob
from tqdm import tqdm
import numpy as np
from pytorch_datasets.dataset import DataSet


class WCVP(DataSet):
    def __init__(self, root):
        self.dataset = []
        for annotation_file in tqdm(glob.glob(root + "*/*.txt"),
                                    ncols=100,
                                    desc="Creating WCVP dataset"
                                    ):
            xyz = open(annotation_file, "r").readlines()[3].split()
            self.dataset.append({
                'image_file': annotation_file.replace('txt', 'jpg'),
                'orientation': np.arctan2(float(xyz[0]), float(xyz[1])) * 180. / np.pi,
                'source': 'WCVP',
            })

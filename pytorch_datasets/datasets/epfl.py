import datetime
from tqdm import tqdm
import pandas as pd
from pytorch_datasets.dataset import DataSet


class EPFL(DataSet):
    def __init__(self, root):
        self.dataset = []
        seq_info = open(root + "tripod-seq.txt", "r").read().split('\n')
        df = pd.DataFrame({
            'total_frames': [int(x) for x in seq_info[1].split()],
            'rotation_frames': [int(x) for x in seq_info[4].split()],
            'front_frame': [int(x) for x in seq_info[5].split()],
            'rotation_dir': [int(x) for x in seq_info[6].split()],
        })
        for seq_idx, row in tqdm(df.iterrows(), ncols=115, desc="Getting EPFL data", total=len(df)):
            times = open(root + "times_{:02d}.txt".format(seq_idx + 1), "r").read().split('\n')
            times = [datetime.datetime.strptime(x, '%Y:%m:%d %H:%M:%S') for x in times[:-1]]
            total_rotation_time = (times[row['rotation_frames'] - 1] - times[0]).total_seconds()
            front_degree_fraction = (times[row['front_frame'] - 1] - times[0]).total_seconds() / total_rotation_time

            for frame in range(0, row['rotation_frames']):
                fraction_through_rotation = (times[frame] - times[0]).total_seconds() / total_rotation_time
                current_orientation = -90 - (-1 * row['rotation_dir'] * (front_degree_fraction - fraction_through_rotation) * 360)

                self.dataset.append({
                    'image_file': root + 'tripod_seq_{:02d}_{:03d}.jpg'.format(seq_idx + 1, frame + 1),
                    'source': 'EPFL',
                    'orientation': current_orientation,
                })

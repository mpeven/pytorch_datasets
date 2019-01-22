import torch
import torchvision
from PIL import Image

class DataSet(torch.utils.data.Dataset):
    """
    Implementation of a PyTorch dataset.

    Only a self.dataset list of dicts needs to be created in a class inheriting from this one.

    Includes functions common to many datasets (image loading, etc.)
        e.g. if 'image_file' is in the dict, this will load the image into an 'image' field.
    """
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        to_return = self.dataset[idx]

        if 'image_file' in to_return:
            to_return['image'] = self.load_image(to_return['image_file'])

        if self.transforms:
            to_return['image'] = self.transforms(to_return['image'])

        return to_return

    def load_image(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f).copy().convert('RGB')
        return img

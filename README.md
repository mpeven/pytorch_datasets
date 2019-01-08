# pytorch_datasets

Repo to keep datasets organized in one location.

Also for keeping loading functions for images, videos, etc.


## Installation

Install into a virtual environment for ease of use.

```
workon <virtualenv name>
cd pytorch_datasets
pip install -e .
```

## Getting Started

Once installed, can import any dataset like so:

```
from pytorch_datasets import EPFL
dset = EPFL.EPFL()
print(dset.__getitem__(0))
```

## Datasets in repo

### EPFL (Multi-View Car Dataset)

- 20 sequences of cars as they rotate by 360 degrees. 2299 images total.
- [Website](https://cvlab.epfl.ch/data/data-pose-index-php/), [Paper](https://infoscience.epfl.ch/record/146798/files/multiview.pdf)

### WCVP (Weizmann Cars ViewPoint)

- Images circling around 22 cars outside. 1530 images total.
- [Website](http://www.wisdom.weizmann.ac.il/~vision/WCVP/), [Paper](http://dx.doi.org/10.1016/j.imavis.2012.09.006)

### ObjectNet3D

- 3D object locations and orientation in images. 100 object categories, 90,127 images, 201,888 objects total in these images.
- [Website](http://cvgl.stanford.edu/projects/objectnet3d/), [Paper](http://cvgl.stanford.edu/papers/xiang_eccv16.pdf)

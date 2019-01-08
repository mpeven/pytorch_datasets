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

### EPFL

### WCVP

### ObjectNet3D

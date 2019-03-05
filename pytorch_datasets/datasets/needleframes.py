'''

Needle Master Dataset

Dataset Information
-------------------

Image frames from demonstrations from the NeedleMaster Android game developped by Chris Paxton. Code for the game is available here https://github.com/cpaxton/needle_master_tools.
The game simulates a suturing task with a needle the user controls with a touch screen, gates to pass the needle through, and tissue. The goal is to pass the needle through all the gates
while inflicting the minimal amount of tissue damage.

NOTE: one dataset element is one image frame/one x timestamp  (to access demonstrations see needlemaster.py)

Location: right now it is on Molly's desktop -- can we move it to the LCSR share?


Dataset Contents:
-----------------
* Environments: There are 20 levels or environments. They contain the positions of the gates and tissue

* Demonstrations: Naming convention: trial_ENVIRONMENT-NUMBER_DEMO-TIMESTAMP. Saved from users playing on their phones. (As of 2/8/19 most demonstrations were from Molly). Demonstrations are a Tx5 array, each row is one timestamp of information. Column 1: time. Column 2: x position of needle. Column 3: y position of needle. Column 4: dX (magnitude of commanded step). Column 5: dTheta (change in needle orientation).

* Images: There is a folder corresponding to each demonstration. Inside the folder each frame from the demonstration is rendered and named with the frame number.

'''
import os
import glob
import math
import click
from tqdm import tqdm
import subprocess
import scipy.io as sio
import pandas as pd
import numpy as np
from PIL import Image
from skimage import io, transform
import torch
import torchvision
from pytorch_datasets.dataset import DataSet
from pdb import set_trace as woah
import random


class NeedleFrames(DataSet):
    """ Can specify train users with the 'train_users' parameter. Otherwise it uses the default """
    def __init__(self, root, environment=None, transforms=None, discrete=False):
        super().__init__(transforms)
        self.root                  = root
        self.video_frames_location = os.path.join(root, 'images/')
        self.discrete              = discrete
        self.environment           = environment

        # Make sure dataset is good to go
        if not self._check_exists():
            raise RuntimeError('Dataset not found at {}'.format(root))


        self.dataset = self.get_all_frames(environment) # NOTE: here we also load the action/needle positions
        self.dataset = self.add_environments(self.dataset)

    def __len__(self):
        """ Return the number of elements in the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx=None):
        """ Return an image from the dataset. If no image is sepecified, return a random frame.
            Right now onlyan image and the position of the needle are returned.
            We can augment this as needed.

            Args:
                idx: the index of the trial in the dataset (int)
                frame_idx: the frame index to extract the video from (int)
        """
        if(idx == None):
            ''' pick a trial '''
            idx = random.randint(0, len(self.dataset))

        trial = self.dataset[idx]['trial'].split('.')[0]
        frame_idx = self.dataset[idx]['frame']

        frame = '{:03d}.png'.format(frame_idx)

        img_name = os.path.join(self.video_frames_location, trial, frame)
        image = io.imread(img_name)

        ''' return needle x y theta position normalized '''
        needle_x     = self.dataset[idx]['x']/float(self.dataset[idx]['environment']['width'])
        needle_y     = self.dataset[idx]['y']/float(self.dataset[idx]['environment']['height'])
        needle_theta = self.dataset[idx]['theta']/(2 * math.pi)

        if(self.discrete):
            needle_x = np.int(np.round(needle_x * 9)) # a way to make 10 classes -- the output can be 0:9
            needle_y = np.int(np.round(needle_y * 9))
            needle_theta = np.int(np.round(needle_theta * 9))

        needle       = np.array([needle_x, needle_y, needle_theta])

        sample = {'image': image[:,:,0:3], 'needle': needle}

        ''' TODO: add transform code '''
        if(self.transforms):
            toPIL = torchvision.transforms.ToPILImage()
            sample['image'] = self.transforms(toPIL(sample['image']))

        return sample

    def get_all_frames(self, environment=None):
        '''
            add one entry for each image in each demonstration
        '''
        dataset = []
        for vid in sorted(os.listdir(os.path.join(self.root, "demonstrations/"))):
            environment_level = vid.split('_')[1]
            if((environment==environment_level) or (environment==None)):
                demo_file = os.path.join(self.root, 'demonstrations', vid)
                df = pd.read_csv(demo_file, names=["t", "x", "y", "theta", "dX", "dtheta"], sep=",")
                t      = df.t.values
                x      = df.x.values
                y      = df.y.values
                theta  = df.theta.values
                dX     = df.dX.values
                dtheta = df.dtheta.values

                for frame in sorted(os.listdir(os.path.join(self.video_frames_location, vid.split('.')[0]))):
                    idx = np.int(frame.split('.')[0])
                    if(idx < len(t)):
                        dataset.append({'trial': vid, 'environment_level': environment_level, 'frame': idx, 't': t[idx], 'x': x[idx], 'y': y[idx], 'theta': theta[idx], \
                        'dX': dX[idx], 'dtheta': dtheta[idx]})
        return dataset

    def add_environments(self, dataset):
        '''
            Add the environment parameters: gate corners
                                            surface points

        '''
        for d in dataset:
            env_file = os.path.join(self.root, 'environments', 'environment_' + d['environment_level'] + '.txt')
            env_fil  = open(env_file, 'r')
            env = load_environment(env_fil)
            d['environment'] = env

        return dataset

    def _check_exists(self):
        return os.path.isdir(self.root) and \
            os.path.isdir(self.root + "/environments") and \
            (os.path.isdir(self.root + "/demonstrations") and os.path.isdir(self.root + "/images"))



''' ------------------------------------------------------
    Utils copied from Chris Paxton to load environment files
    ------------------------------------------------------ '''
def safe_load_line(name, fil):
    l = fil.readline()[:-1].split(': ')
    assert(l[0] == name)

    return l[1].split(',')

'''
Load an environment file.
'''
def load_environment(fil):
    Environment = {}
    D = safe_load_line('Dimensions',fil)
    Environment['height'] = int(D[1])
    Environment['width']  = int(D[0])

    D = safe_load_line('Gates',fil)
    Environment['ngates'] = int(D[0])

    Gates = []
    for _ in range(Environment['ngates']):
        gate = {}
        gate = load_gate(gate, fil, Environment['height'], Environment['width'])
        Gates.append(gate)
    Environment['Gates'] = Gates

    D = safe_load_line('Surfaces', fil)
    Environment['nsurfaces'] = int(D[0])

    Surfaces = []
    for _ in range(Environment['nsurfaces']):
        s = {}
        s = load_surface(s, fil, Environment['height'])
        Surfaces.append(s)
    Environment['Surfaces'] = Surfaces
    return Environment

'''
Load Gate from file
'''
def load_gate(gate, fil, height, width):

    pos = safe_load_line('GatePos',fil)
    cornersx = safe_load_line('GateX',fil)
    cornersy = safe_load_line('GateY',fil)
    topx = safe_load_line('TopX',fil)
    topy = safe_load_line('TopY',fil)
    bottomx = safe_load_line('BottomX',fil)
    bottomy = safe_load_line('BottomY',fil)

    gate['x'] = width*float(pos[0])
    gate['y'] = height*float(pos[1])
    gate['w'] = float(pos[2])

    gate['top']     = np.zeros((len(topx), 2))
    gate['bottom']  = np.zeros((len(topx), 2))
    gate['corners'] = np.zeros((len(topx), 2))

    gate['top'][:,0]     = [float(x) for x in topx]
    gate['top'][:,1]     = [float(y) for y in topy]
    gate['bottom'][:,0]  = [float(x) for x in bottomx]
    gate['bottom'][:,1]  = [float(y) for y in bottomy]
    gate['corners'][:,0] = [float(x) for x in cornersx]
    gate['corners'][:,1] = [float(y) for y in cornersy]

    # apply corrections to make sure the gates are oriented right
    gate['w'] *= -1
    if gate['w'] < 0:
        gate['w'] = gate['w'] + (np.pi * 2)
    if gate['w'] > np.pi:
        gate['w'] -= np.pi
        gate['top'] = np.squeeze(gate['top'][np.ix_([2,3,0,1]),:2])
        gate['bottom'] = np.squeeze(gate['bottom'][np.ix_([2,3,0,1]),:2])
        gate['corners'] = np.squeeze(gate['corners'][np.ix_([2,3,0,1]),:2])

    gate['w'] -= np.pi / 2

    avgtopy = np.mean(gate['top'][:,1])
    avgbottomy = np.mean(gate['bottom'][:,1])

    # flip top and bottom if necessary
    if avgtopy < avgbottomy:
        tmp = gate['top']
        gate['top'] = gate['bottom']
        gate['bottom'] = tmp

    return gate

'''
Load surface from file at the current position
'''
def load_surface(surface, fil, height):
    isdeep = safe_load_line('IsDeepTissue',fil)

    sx = [float(x) for x in safe_load_line('SurfaceX',fil)]
    sy = [float(x) for x in safe_load_line('SurfaceY',fil)]
    surface['corners'] = np.array([sx,sy]).transpose()
    surface['corners'][:,1] = height - surface['corners'][:,1]
    surface['deep'] = isdeep[0] == 'true'

    return surface

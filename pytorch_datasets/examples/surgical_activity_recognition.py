"""
Example dataset use. Single image activity recognition.
Classifies 74.61% of frames correctly on the test splitself.
"""

import pytorch_datasets
from tqdm import tqdm
import signal
import sys
import numpy as np
import torch
import torchvision
import multiprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))

class Model(torch.nn.Module):
    """ Example Model. ResNet18 with a prediction layer on top. """
    def __init__(self, num_labels):
        super(Model, self).__init__()
        self.base_model = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)
        self.preds = torch.nn.Linear(base_model_fc_size, num_labels)

    def forward(self, images):
        im_features = self.base_model(images)
        preds = self.preds(im_features.squeeze())
        return preds

def epoch(train_mode, description, model, dataloader, optimizer=None, loss_func=None):
    """ Train, validation, or test epoch """
    # Create dataset iterator
    iterator = tqdm(dataloader, ncols=100, desc=description)

    # Turn off batch norm, etc. during testing/validation
    model = model.train(train_mode)

    # Data to print
    running_losses, running_labels, running_predict = [], [], []

    # Loop over all data
    with torch.set_grad_enabled(train_mode):
        for data in iterator:
            outputs = model(data['image'].to(DEVICE)) # Forward pass

            if train_mode:
                optimizer.zero_grad() # Zero out gradients
                loss = loss_func(outputs, data['maneuver_index'].to(DEVICE))
                running_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # Update labels and predictions
            running_labels.append(data["maneuver_index"].numpy().copy())
            prediction = outputs.cpu().detach().numpy()
            running_predict.append(np.argmax(prediction, 1))

            # Update accuracy
            accuracy = 100.0 * np.mean(np.equal(np.concatenate(running_labels), np.concatenate(running_predict)))
            info_to_show = {'Acc': "{:.4f}".format(accuracy)}
            if train_mode:
                info_to_show['Loss'] = "{:.5f}".format(np.mean(running_losses))
            iterator.set_postfix(info_to_show)

def modify_dataset(dataset, maneuver_labels):
    ''' Convert from per-trial to per-frame and remove any "No_Maneuver" frames '''
    updated_dataset = []
    for trial in dataset:
        # Needs to have a maneuver file
        if trial['contains_maneuver_annotations'] == False:
            continue
        for idx, frame in enumerate(trial['frames']):
            maneuver_name = trial['maneuver_names'][idx]
            if maneuver_name not in maneuver_labels:
                continue
            updated_dataset.append({
                'image_file': '/hdd/Datasets/MISTIC/video_frames/{}_right/{:05d}.jpg'.format(trial['trial'], idx),
                'maneuver_name': maneuver_name,
                'maneuver_index': maneuver_labels.index(maneuver_name)
            })
    return updated_dataset


def main():
    # Create image tranforms
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ColorJitter(.2, .2, .2, .2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    dataset_train = pytorch_datasets.MISTIC('/hdd/Datasets/MISTIC', train_split='train', transforms=transforms_train)
    dataset_val   = pytorch_datasets.MISTIC('/hdd/Datasets/MISTIC', train_split='val',   transforms=transforms_test)
    dataset_test  = pytorch_datasets.MISTIC('/hdd/Datasets/MISTIC', train_split='test',  transforms=transforms_test)

    # Modify datasets
    maneuver_labels = [
        'Suture_Throw', 'Grasp_Pull_Run_Suture', 'Inter_Maneuver_Segment', 'Knot_PSM1_1Loop_ACW',
        'Knot_PSM1_1Loop_CW', 'Knot_PSM1_2Loop_CW', 'Knot_PSM2_1Loop_CW', 'Knot_PSM2_1Loop_ACW',
        'Knot_PSM2_2Loop_CW', 'Knot_PSM2_2Loop_ACW', 'Undo_Maneuver',
    ]
    dataset_train.dataset = modify_dataset(dataset_train.dataset, maneuver_labels)
    dataset_val.dataset   = modify_dataset(dataset_val.dataset,   maneuver_labels)
    dataset_test.dataset  = modify_dataset(dataset_test.dataset,  maneuver_labels)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True,  batch_size=256, num_workers=multiprocessing.cpu_count())
    dataloader_val   = torch.utils.data.DataLoader(dataset_val,   shuffle=False, batch_size=256, num_workers=multiprocessing.cpu_count())
    dataloader_test  = torch.utils.data.DataLoader(dataset_test,  shuffle=False, batch_size=256, num_workers=multiprocessing.cpu_count())

    # Create Model
    num_maneuvers = len(maneuver_labels)
    model = Model(num_maneuvers).to(DEVICE)

    # Create loss function and optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)

    # Train + Val
    num_epochs = 20
    for epoch_idx in range(num_epochs):
        print("Epoch {}/{}".format(epoch_idx+1, num_epochs))
        epoch(True, "Training", model, dataloader_train, optimizer, loss_func)
        epoch(False, "Validating", model, dataloader_val)

    # Test
    epoch(False, "Testing", model, dataloader_test)

if __name__ == '__main__':
    main()

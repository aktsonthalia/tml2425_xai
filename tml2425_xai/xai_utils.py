# The dummy feature attribution explanation is a centered gaussian map
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd

from .cub_dataset import Cub2011

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_cub_test():
    torch.manual_seed(2022) # Set seed for reproducibility
    _IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
    _IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

    def set_data_targets(dataset):
        #required for this exercise
        data = []
        targets = []
        for img, label in dataset:
            data.append(img)
            targets.append(label)

        dataset.data = torch.stack(data)
        dataset.targets = torch.tensor(targets)

    # Load the dataset with a center crop 
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE) 
                                    ])

    cub_test = Cub2011(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    set_data_targets(cub_test)

    #subsample cub_test to 1000 instances, 5 images per class
    grouped_cub = cub_test.meta_data.groupby('target')
    sample_index = pd.concat([data.sample(n=5, random_state=2022) for _, data in grouped_cub])
    new_indices = []
    for sample_img_id in sample_index['img_id']:
        new_indices.append(np.where(cub_test.meta_data['img_id'] == sample_img_id)[0][0])

    cub_test = torch.utils.data.Subset(cub_test, sample_index)
    cub_test.indices = new_indices
    set_data_targets(cub_test)
    return cub_test


def centered_gaussian(xres, yres):
    """Baseline centered Gaussian heatmap
    """
    # adapted from https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python
    # create gaussian kernel
    mean = (0,0)
    cov = np.eye(2)
    gauss = multivariate_normal(mean, cov)

    xlim = (-3, 3)
    ylim = (-3, 3)

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = gauss.pdf(xxyy) + gauss.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres,yres))
    return img
    
    
# Evaluate the model test set accuracy
def model_accuracy(model, dataset):
    data_loader = DataLoader(dataset,
                            batch_size=8)
    correct_total = 0

    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device) 

        y_pred = model(x_batch)
        y_pred_max = torch.argmax(y_pred, dim=1)

        correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()
    accuracy = correct_total / len(dataset)
    return accuracy

def show_sample(dataset, sample_idx):
    """Visualization method of an image
    
    :param dataset: dataset split
    :param sample_idx: id of the sample to be shown (int)
    """
    figure = plt.figure(figsize=(3,3))
    img, label = dataset[sample_idx]
    #just for displaying, without normalization
    unnormalize_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),
                                                ])
    img = unnormalize_transform(img)
    img = img.permute(1,2,0)  # RGB channel to the back
    plt.title(f'Ground truth label: {label}')
    plt.axis("off")
    plt.imshow(img.squeeze())
    plt.show()


def show_attribution_overlay(dataset, sample_idx, attribution_map, title):
    """Visualization of an attribution explanation overlayed to the image it's explaining

    :param dataset: dataset split
    :param sample_idx: id of the sample to be shown (int)
    :param attribution_map: explanation of size (224,224)
    :param title: title string
    """
    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    x = np.arange(0., 224.0, dx)
    y = np.arange(0., 224.0, dy)
    X, Y = np.meshgrid(x, y)
    
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    
    img, label = dataset[sample_idx]
    #just for displaying, without normalization
    unnormalize_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),
                                                ])
    img = unnormalize_transform(img)
    img = img.permute(1,2,0) # RGB channel to the back
    im1 = plt.imshow(img, interpolation='nearest',
                 extent=extent)

    im2 = plt.imshow(attribution_map, cmap=plt.cm.viridis, alpha=.8, interpolation='bilinear',
                    extent=extent)
    plt.axis('off')
    plt.title(title)

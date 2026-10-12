# %%
import sys
import os
sys.path.append(os.path.abspath('..'))  # Arregla el "ModuleNotFoundError"

import torch
from spectralwaste_segmentation import models, datasets, utils
import matplotlib.pyplot as plt

# %%
# Arregla el "FileNotFoundError" (apuntando a '../data')
data_rgb = datasets.SpectralWasteSegmentation('../data', split='train', input_mode='rgb', target_mode='labels_rgb', target_type='')
data_hyper = datasets.SpectralWasteSegmentation('../data', split='train', input_mode='hyper', target_mode='labels_hyper_lt', target_type='')

# %%
fig, axs = plt.subplots(3, 8, figsize=(20, 7.5), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

for ax in axs.flatten():
    ax.axis('off')

for i, idx in enumerate([0, 479, 352, 200, 107, 305, 143, 6]):
    rgb, labels_rgb = data_rgb[idx]
    hyper, labels_hyper = data_hyper[idx]

    axs[0, i].imshow(rgb.permute(1, 2, 0))
    utils.plot_labels(rgb.permute(1, 2, 0), labels_rgb, data_rgb.palette, axs[1, i])
    utils.plot_labels(utils.false_color(hyper.permute(1, 2, 0).numpy()), labels_hyper, data_rgb.palette, axs[2, i])
# %%

import os
import sys

# 1. Ruta raíz del proyecto TFG
ROOT_DIR = '/content/drive/MyDrive/TFG'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import matplotlib.pyplot as plt
from spectralwaste_segmentation import datasets, models, utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# 2. Rutas del dataset y del checkpoint de VM-UNet
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CHECKPOINT_PATH = os.path.join(
    ROOT_DIR, 
    'checkpoints', 
    'vm_unet_multimodal.rgb,hyper_pca3.labels_rgb.2d27.last.pth'
)

# 3. Cargar Dataset de TEST (Se pasa una lista ['rgb', 'hyper_pca3'] en lugar del string unido)
data_test = datasets.SpectralWasteSegmentation(
    DATA_DIR,
    split='test',
    input_mode=['rgb', 'hyper_pca3'],
    target_mode='labels_rgb',
    target_type=''
)

num_classes = len(data_test.classes)

# 4. Cargar modelo VM-UNet e importar pesos
try:
    model = models.create_model('vm_unet_multimodal', [3, 3], num_classes)
except AttributeError:
    model = models.get_model('vm_unet_multimodal', in_channels=6, num_classes=num_classes)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(state_dict)
elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
print("¡Modelo VM-UNet cargado con éxito!")

# 5. Generar figura comparativa
indices = [0, 6, 107, 143, 200]
fig, axs = plt.subplots(
    len(indices), 
    3, 
    figsize=(10, 3.2 * len(indices)), 
    gridspec_kw={'hspace': 0.08, 'wspace': 0.05}
)

axs[0, 0].set_title('RGB (Entrada)', fontsize=12, fontweight='bold', pad=10)
axs[0, 1].set_title('Ground Truth (GT)', fontsize=12, fontweight='bold', pad=10)
axs[0, 2].set_title('Predicción VM-UNet', fontsize=12, fontweight='bold', pad=10)

for i, idx in enumerate(indices):
    inputs, labels_rgb = data_test[idx]

    if isinstance(inputs, list):
        rgb_img = inputs[0].permute(1, 2, 0)
        x = [t.unsqueeze(0).to(device) for t in inputs]
    else:
        rgb_img = inputs[:3, :, :].permute(1, 2, 0)
        x = inputs.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Columna 1: Imagen RGB
    axs[i, 0].imshow(rgb_img)
    axs[i, 0].axis('off')

    # Columna 2: Ground Truth
    utils.plot_labels(rgb_img, labels_rgb, data_test.palette, axs[i, 1])
    axs[i, 1].axis('off')

    # Columna 3: Predicción VM-UNet
    utils.plot_labels(rgb_img, pred_mask, data_test.palette, axs[i, 2])
    axs[i, 2].axis('off')

plt.tight_layout()
output_fig_path = os.path.join(ROOT_DIR, 'figura_cualitativa_vm_unet.png')
plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
print(f"¡Figura guardada exitosamente en: {output_fig_path}!")

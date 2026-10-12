# %%
import sys
import os
import argparse
sys.path.append(os.path.abspath('..'))

import torch
import matplotlib.pyplot as plt

from spectralwaste_segmentation import models, datasets, utils

# Permitir deserialización segura de argumentos en PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

# %% Lista de los 12 experimentos
experiments = [
    'mininet.rgb.labels_rgb.b86a',
    'mininet.hyper.labels_hyper_lt.009d',
    'mininet.hyper_pca3.labels_hyper_lt.3b56',
    'mininet_multimodal.rgb,hyper.labels_rgb.3154',
    'mininet_multimodal.rgb,hyper_pca3.labels_rgb.8fad',

    'segformer_b0.rgb.labels_rgb.57a9',
    'segformer_b0.hyper.labels_hyper_lt.ead7',
    'segformer_b0.hyper_pca3.labels_hyper_lt.ee6f',
    'segformer_b0_multimodal.rgb,hyper.labels_rgb.451d',
    'segformer_b0_multimodal.rgb,hyper_pca3.labels_rgb.4cff',

    'cmx_b0.rgb,hyper.labels_rgb.b5c4',
    'cmx_b0.rgb,hyper_pca3.labels_rgb.55b1',

    
]

# %% Selección de muestras del conjunto de test
input_idxs = [0, 165, 79, 166, 118]

# Crear figura para 5 filas x 14 columnas (RGB + GT + 12 Experimentos)
fig, axs = plt.subplots(
    len(input_idxs), 
    len(experiments) + 2, 
    figsize=(36, 10), 
    squeeze=False, 
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
)

for i, name in enumerate(experiments):
    checkpoint_path = f'../checkpoints/{name}.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    print(f"[{i+1}/{len(experiments)}] Procesando: {name}")

    # Cargar dataset según los modos del experimento actual
    data = datasets.SpectralWasteSegmentation(
        '../data', 
        split='test', 
        input_mode=args.input_mode, 
        target_mode=args.target_mode, 
        target_type=''
    )

    # Recrear modelo y cargar pesos
    model = models.create_model(args.model, data.num_channels, data.num_classes)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for j, idx in enumerate(input_idxs):
        with torch.inference_mode():
            input_data, target = data[idx][0], data[idx][1]

            # Inferencia adaptativa (Unimodal vs Multimodal)
            if isinstance(input_data, list):
                input_tensor = [inp.unsqueeze(0) for inp in input_data]
                try:
                    out = model(*input_tensor)
                except TypeError:
                    out = model(input_tensor)
            else:
                input_tensor = input_data.unsqueeze(0)
                out = model(input_tensor)

            # Normalización del formato de salida del modelo
            if isinstance(out, (tuple, list)):
                out = out[0]
            elif hasattr(out, 'logits'):
                out = out.logits

            # Máscara predicha (clase con mayor probabilidad por píxel)
            out = out.argmax(dim=1).squeeze()

        # Dibujar imagen RGB de entrada y Ground Truth (solo en la primera iteración)
        if i == 0:
            rgb_img = input_data[0] if isinstance(input_data, list) else input_data
            rgb_np = rgb_img.permute(1, 2, 0).cpu().numpy()
            
            # Normalizar entre 0 y 1 si la imagen viene en float fuera de rango
            if rgb_np.max() > 1.0:
                rgb_np = rgb_np / 255.0
            rgb_np = rgb_np.clip(0, 1)

            axs[j, 0].imshow(rgb_np)
            axs[j, 0].axis('off')
            
            axs[j, 1].imshow(utils.get_color_labels(target, data.palette), interpolation='none')
            axs[j, 1].axis('off')

        # Dibujar predicción del experimento actual
        axs[j, i+2].imshow(utils.get_color_labels(out, data.palette), interpolation='none')
        axs[j, i+2].axis('off')

    # Títulos de las columnas
    if i == 0:
        axs[0, 0].set_title("RGB", fontsize=8, fontweight='bold')
        axs[0, 1].set_title("Ground Truth", fontsize=8, fontweight='bold')

    title_input = ','.join(args.input_mode) if isinstance(args.input_mode, list) else args.input_mode
    axs[0, i+2].set_title(f"{args.model}\n({title_input})", fontsize=6.5)

plt.tight_layout()
plt.show()
# %%

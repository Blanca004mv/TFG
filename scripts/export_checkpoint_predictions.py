import sys
import os
import argparse
import random
import torch
import matplotlib.pyplot as plt

# Permitir deserialización segura de argumentos en PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

from spectralwaste_segmentation import models, datasets, utils

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exportar grid cualitativo (RGB, Ground Truth, Predicción) a partir de un checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path", 
        type=str, 
        required=True, 
        help="Ruta absoluta o relativa al archivo .pth / .pt del checkpoint."
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="../data", 
        help="Ruta a la carpeta base del dataset SpectralWaste."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/content/drive/MyDrive/TFG/qualitative_results", 
        help="Carpeta donde se guardarán las imágenes generadas."
    )
    parser.add_argument(
        "--input-idxs", 
        type=int, 
        nargs="+", 
        default=[0, 165, 79, 166, 118], 
        help="Índices de las muestras de test a visualizar (separados por espacio)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Dispositivo para ejecutar la inferencia ('cpu' o 'cuda')."
    )
    return parser.parse_args()

def main():
    args_cli = parse_args()

    if not os.path.exists(args_cli.checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint en: {args_cli.checkpoint_path}")

    print(f"Cargando checkpoint desde: {args_cli.checkpoint_path}")
    try:
        checkpoint = torch.load(args_cli.checkpoint_path, map_location=args_cli.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args_cli.checkpoint_path, map_location=args_cli.device)

    # Extraer argumentos guardados durante el entrenamiento
    model_args = checkpoint.get('args', None)

    if model_args is None:
        raise ValueError(
            "El checkpoint no contiene los metadatos 'args' de entrenamiento. "
            "Asegúrate de que el checkpoint incluye el diccionario 'args'."
        )

    # Identificar arquitectura e input_mode del checkpoint
    model_name = getattr(model_args, 'model', getattr(model_args, 'model_name', 'modelo_desconocido'))
    input_mode = getattr(model_args, 'input_mode', ['rgb'])
    target_mode = getattr(model_args, 'target_mode', 'labels_rgb')

    print(f"Configuración detectada en el checkpoint:")
    print(f" -> Modelo: {model_name}")
    print(f" -> Modos de Entrada: {input_mode}")
    print(f" -> Modo de Objetivo: {target_mode}")

    # 1. Cargar Dataset de Test
    print(f"\nCargando dataset de Test desde: {args_cli.data_path}")
    data = datasets.SpectralWasteSegmentation(
        args_cli.data_path, 
        split='test', 
        input_mode=input_mode, 
        target_mode=target_mode, 
        target_type=''
    )

    # 2. Recrear e Instanciar Modelo
    print(f"Instanciando modelo '{model_name}'...")
    model = models.create_model(model_name, data.num_channels, data.num_classes)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(args_cli.device)
    model.eval()

    # 3. Crear la Figura Comparativa
    num_samples = len(args_cli.input_idxs)
    fig, axs = plt.subplots(
        num_samples, 
        3, 
        figsize=(8, 2.2 * num_samples), 
        squeeze=False, 
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
    )

    with torch.no_grad():
        for j, idx in enumerate(args_cli.input_idxs):
            input_data, target = data[idx][0], data[idx][1]

            # Inferencia adaptativa según la estructura de entrada
            if isinstance(input_data, list):
                input_tensor = [inp.unsqueeze(0).to(args_cli.device) for inp in input_data]
                try:
                    out = model(*input_tensor)
                except TypeError:
                    out = model(input_tensor)
            else:
                input_tensor = input_data.unsqueeze(0).to(args_cli.device)
                out = model(input_tensor)

            # Normalizar formato de salida de los logits
            if isinstance(out, (tuple, list)):
                out = out[0]
            elif hasattr(out, 'logits'):
                out = out.logits

            # Obtener máscara con argmax
            pred_mask = out.argmax(dim=1).squeeze().cpu()

            # 1. Columna 0: Imagen RGB
            rgb_img = input_data[0] if isinstance(input_data, list) else input_data
            rgb_np = rgb_img.permute(1, 2, 0).cpu().numpy()
            if rgb_np.max() > 1.0:
                rgb_np = rgb_np / 255.0
            rgb_np = rgb_np.clip(0, 1)

            axs[j, 0].imshow(rgb_np)
            axs[j, 0].axis('off')

            # 2. Columna 1: Ground Truth
            axs[j, 1].imshow(utils.get_color_labels(target, data.palette), interpolation='none')
            axs[j, 1].axis('off')

            # 3. Columna 2: Predicción del Modelo
            axs[j, 2].imshow(utils.get_color_labels(pred_mask, data.palette), interpolation='none')
            axs[j, 2].axis('off')

    # Títulos superiores
    axs[0, 0].set_title("RGB", fontsize=10, fontweight='bold')
    axs[0, 1].set_title("Ground Truth", fontsize=10, fontweight='bold')

    title_input = ','.join(input_mode) if isinstance(input_mode, list) else input_mode
    axs[0, 2].set_title(f"{model_name}\n({title_input})", fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Definir subcarpeta de salida según el modelo y guardar imagen
    save_dir = os.path.join(args_cli.output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_basename = os.path.splitext(os.path.basename(args_cli.checkpoint_path))[0]
    output_file = os.path.join(save_dir, f"{checkpoint_basename}_qualitative_grid.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n¡Proceso finalizado! Resultado cualitativo guardado en:\n -> {output_file}")

if __name__ == "__main__":
    main()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def white_patch(img: np.ndarray):
    img_float = img.astype(np.float32)
    max_channel_value = np.max(img_float, axis=(0, 1))
    cantidad_de_desviaciones = 1.5
    correction_factors = np.ones(3, dtype=np.float32)

    for i in range(len(max_channel_value)):
        if max_channel_value[i] < 1:
            correction_factors[i] = 255 / (np.mean(
                img_float[:, :, i]) - cantidad_de_desviaciones*np.std(img_float[:, :, i]))
        elif max_channel_value[i] > 254:
            correction_factors[i] = 255 / (np.mean(
                img_float[:, :, i]) + cantidad_de_desviaciones*np.std(img_float[:, :, i]))
        else:
            correction_factors[i] = 255.0 / \
                np.clip(max_channel_value[i], 1, 254)

    corrected_img = img_float * correction_factors
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

    return corrected_img


def white_patch_intelligent(img: np.ndarray, method='percentile', percentile=99.5, edge_threshold=0.1):
    img_float = img.astype(np.float32)

    if method == 'percentile':
        reference_values = np.percentile(img_float, percentile, axis=(0, 1))

    elif method == 'edge_based':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        reference_values = []
        for i in range(3):
            channel_values = img_float[:, :, i][edges_dilated > 0]
            if len(channel_values) > 0:
                reference_values.append(np.percentile(channel_values, 95))
            else:
                reference_values.append(np.max(img_float[:, :, i]))
        reference_values = np.array(reference_values)

    elif method == 'iterative':
        reference_values = np.zeros(3)
        for i in range(3):
            channel = img_float[:, :, i].copy()

            saturated_pixels = channel[channel > 240]

            if len(saturated_pixels) == 0:
                reference_values[i] = np.max(channel)
            else:
                for iteration in range(3):
                    valid_pixels = channel[channel < 240]
                    if len(valid_pixels) > 0:
                        threshold = np.percentile(valid_pixels, 98)
                        channel = valid_pixels[valid_pixels <= threshold]
                    else:
                        break
                reference_values[i] = np.max(
                    channel) if len(channel) > 0 else 255

    elif method == 'robust_max':
        reference_values = []
        for i in range(3):
            channel = img_float[:, :, i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)

            if std_val < 10:
                reference_values.append(np.percentile(channel, 99))
            elif std_val > 50:
                reference_values.append(mean_val + 2.0 * std_val)
            else:
                reference_values.append(mean_val + 1.5 * std_val)
        reference_values = np.array(reference_values)

    reference_values = np.clip(reference_values, 1, 254)
    correction_factors = 255.0 / reference_values
    corrected_img = img_float * correction_factors
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

    return corrected_img


def white_patch_adaptive(img: np.ndarray):
    img_float = img.astype(np.float32)

    mean_brightness = np.mean(img_float)
    std_brightness = np.std(img_float)

    saturated_pixels = np.sum(img_float > 250) / img_float.size

    if saturated_pixels > 0.01:
        method = 'iterative'
    elif std_brightness < 20:
        method = 'percentile'
    elif mean_brightness < 80:
        method = 'robust_max'
    else:
        method = 'edge_based'

    return white_patch_intelligent(img, method=method)


BASE_IMG_FOLDERS = Path("Material_TPs/TP1/white_patch")
RESULTS_FOLDER = Path("results")
RESULTS_FOLDER.mkdir(exist_ok=True)
image_paths = list(BASE_IMG_FOLDERS.glob("*.png")) + \
    list(BASE_IMG_FOLDERS.glob("*.jpg"))
n_images = len(image_paths)

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    corrected_img = white_patch_adaptive(img.copy())
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corrected_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original: {img_path.name}")
    axes[0].axis('off')

    axes[1].imshow(corrected_rgb)
    axes[1].set_title(f"White Patch: {img_path.name}")
    axes[1].axis('off')

    plt.tight_layout()

    result_filename = RESULTS_FOLDER / f"{img_path.stem}_comparison.png"
    plt.savefig(result_filename, dpi=300, bbox_inches='tight')
    # plt.show()

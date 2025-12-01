import cv2
import numpy as np

def unsharp_masking(img: np.ndarray, k: float) -> np.ndarray:
    """
    Aplica Unsharp Masking para mejorar la nitidez de una imagen.

    Args:
        img (np.ndarray): Imagen de entrada en formato BGR (MxNx3).
        k (float): Factor de enfocado. Valores mayores producen mayor nitidez.

    Returns:
        np.ndarray: Imagen enfocada mediante Unsharp Masking.
    """

    # Suavizado con filtro Gaussiano
    # Kernel 7x7 y sigma 0.5 

    gauss = cv2.GaussianBlur(img, (7, 7), 0.5)

    # Aplicación de Unsharp Masking usando addWeighted:
    # sharpened = img * (k + 1) + gauss * (-k)
    
    img_sharp = cv2.addWeighted(img, k + 1, gauss, -k, 0)

    return img_sharp



import cv2
import numpy as np

def lv_laplacian_variance(img):
    """
    Calcula la métrica de enfoque LV (Laplacian Variance) para una imagen.

    Args:
        img (np.ndarray): Imagen en escala de grises o color (MxN).

    Returns:
        float: Medición de calidad de imagen LV. Valores mayores indican mayor nitidez.
    """
  
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_f = np.float64(img)

    lap = cv2.Laplacian(img_f, ddepth=cv2.CV_64F)

    mean, std = cv2.meanStdDev(lap)
    variance = std[0][0] ** 2

    return float(variance)

def lv_laplacian_variance_video(video_path):
    """
    Calcula la métrica LV para cada frame de un video usando la varianza del Laplaciano.

    Args:
        video_path (str): Ruta del archivo de video.

    Returns:
        lv_values:Valores LV de cada frame.
        frame_numbers: Índices de los frames.
        lv_max (float: Valor máximo LV en el video.
        frame_max: Índice del frame con LV máximo.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    lv_values, frame_numbers = [], []
    frame_idx = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        lv_values.append(lv_laplacian_variance(frame))
        frame_numbers.append(frame_idx)
        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()

    if lv_values:
        lv_max = max(lv_values)
        frame_max = lv_values.index(lv_max)
    else:
        lv_max = None
        frame_max = None

    return lv_values, frame_numbers, lv_max, frame_max

def lv_laplacian_variance_roi(img, roi = 0.05):
    """
    Calcula la métrica LV en una región central de la imagen.

    Args:
        img: Imagen en escala de grises o color (MxN).
        roi: Proporción del área de la ROI respecto a la imagen (0-1).

    Returns:
        lv: Medición de calidad de la región.
        rect: Coordenadas del rectángulo ROI (x1, y1, x2, y2).
    """
    h, w = img.shape[:2]
    total_area = h * w
    roi_area = int(total_area * roi)
    roi_side = int(np.sqrt(roi_area))

    cx, cy = w // 2, h // 2
    x1 = max(cx - roi_side // 2, 0)
    y1 = max(cy - roi_side // 2, 0)
    x2 = min(cx + roi_side // 2, w)
    y2 = min(cy + roi_side // 2, h)

    roi_img = img[y1:y2, x1:x2]
    lv = lv_laplacian_variance(roi_img)

    return lv, (x1, y1, x2, y2)


def lv_laplacian_variance_roi_video(video_path, roi = 0.05):
    """
    Calcula la métrica LV en la región central para cada frame de un video.

    Args:
        video_path: Ruta del video.
        roi: Proporción del área de la ROI respecto a la imagen (0-1).

    Returns:
        lv_values: Valores LV por frame.
        frame_numbers: Índices de frames.
        lv_max: Valor máximo LV en el video.
        frame_max: Índice del frame con LV máximo.
        rect: Rectángulo de la ROI.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    lv_values, frame_numbers = [], []
    frame_idx = 0
    rect = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        lv, rect = lv_laplacian_variance_roi(frame, roi)
        lv_values.append(lv)
        frame_numbers.append(frame_idx)
        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()

    if lv_values:
        lv_max = max(lv_values)
        frame_max = lv_values.index(lv_max)
    else:
        lv_max = None
        frame_max = None

    return lv_values, frame_numbers, lv_max, frame_max, rect

def lv_laplacian_variance_grid(img, N: int = 3, M: int = 3):
    """
    Calcula LV en una grilla NxM de cuadrados centrados en la imagen.

    Cada cuadrado tiene separación = 0.5 * lado y borde desde la imagen = 0.5 * lado.

    Args:
        img: Imagen de entrada (MxN).
        N: Filas de la grilla.
        M: Columnas de la grilla.

    Returns:
        lv_matrix: Matriz NxM con valores LV por bloque.
        rect_matrix: Matriz NxM con coordenadas de cada bloque (x1, y1, x2, y2).
    """
    H, W = img.shape[:2]
    S_w = 2 * W / (3 * M + 1)
    S_h = 2 * H / (3 * N + 1)
    S = int(min(S_w, S_h))

    if S < 2:
        raise ValueError("La grilla queda demasiado pequeña o no entra.")

    gap = S // 2
    border = S // 2

    total_w = S * (3 * M + 1) / 2
    total_h = S * (3 * N + 1) / 2

    offset_x = int((W - total_w) / 2 + border)
    offset_y = int((H - total_h) / 2 + border)

    lv_matrix = np.zeros((N, M), dtype=float)
    rect_matrix = np.empty((N, M), dtype=object)

    for i in range(N):
        for j in range(M):
            x1 = offset_x + j * (S + gap)
            y1 = offset_y + i * (S + gap)
            x2 = x1 + S
            y2 = y1 + S

            rect_matrix[i, j] = (int(x1), int(y1), int(x2), int(y2))
            block = img[y1:y2, x1:x2]
            lv_matrix[i, j] = lv_laplacian_variance(block)

    return lv_matrix, rect_matrix


def lv_laplacian_variance_grid_video(video_path: str, N: int = 3, M: int = 3):
    """
    Calcula LV usando una grilla NxM de bloques para cada frame de un video.

    Args:
        video_path: Ruta del video.
        N: Filas de la grilla.
        M: Columnas de la grilla.

    Returns:
        lv_values: LV máximo de cada frame.
        frame_numbers: Índices de frames.
        lv_max: LV máximo global del video.
        frame_max: Frame donde ocurre el LV máximo global.
        rect_max: Rectángulo del LV máximo global.
        rect_matrices: Lista de matrices NxM con rectángulos de cada frame.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    lv_values, frame_numbers = [], []
    lv_matrices, rect_matrices = [], []
    frame_idx = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        lv_matrix, rect_matrix = lv_laplacian_variance_grid(frame, N, M)
        lv_matrices.append(lv_matrix)
        rect_matrices.append(rect_matrix)

        lv_values.append(lv_matrix.max())
        frame_numbers.append(frame_idx)
        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()

    if not lv_values:
        return [], [], None, None, None, rect_matrices

    lv_max = max(lv_values)
    frame_max = lv_values.index(lv_max)
    lv_matrix_target = lv_matrices[frame_max]

    i, j = np.unravel_index(np.argmax(lv_matrix_target), lv_matrix_target.shape)
    rect_max = rect_matrices[frame_max][i, j]

    return lv_values, frame_numbers, lv_max, frame_max, rect_max, rect_matrices
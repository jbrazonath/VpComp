import cv2
import numpy as np

def fm_sharpness(img: np.ndarray) -> float:
    """
    Calcula la métrica de enfoque FM (Frequency-domain Measure) para una imagen.

    Args:
        img (np.ndarray): Imagen en escala de grises o color (MxN).

    Returns:
        float: Medición de calidad de imagen FM. Valores mayores indican mayor nitidez.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_f = np.float32(img)
    F = np.fft.fft2(img_f)
    Fc = np.fft.fftshift(F)
    AF = np.abs(Fc)
    M_val = AF.max()

    if M_val == 0:
        return 0.0

    threshold = M_val / 1000.0
    TH = np.count_nonzero(AF > threshold)

    m, n = img.shape[:2]
    FM = TH / (m * n)

    return FM


def fm_sharpness_video(video_path: str):
    """
    Calcula la métrica FM para cada frame de un video.

    Args:
        video_path (str): Ruta del archivo de video.

    Returns:
        fm_values (list[float]): Valores FM de cada frame.
        frame_numbers (list[int]): Índices de los frames.
        fm_max (float | None): Valor máximo FM en el video.
        frame_max (int | None): Índice del frame con FM máximo.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fm_values, frame_numbers = [], []
    frame_idx = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        fm_values.append(fm_sharpness(frame))
        frame_numbers.append(frame_idx)
        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()

    if fm_values:
        fm_max = max(fm_values)
        frame_max = fm_values.index(fm_max)
    else:
        fm_max = None
        frame_max = None

    return fm_values, frame_numbers, fm_max, frame_max


def fm_sharpness_roi(img: np.ndarray, roi: float = 0.05):
    """
    Calcula la métrica FM en una región central de la imagen.

    Args:
        img (np.ndarray): Imagen en escala de grises o color (MxN).
        roi (float): Proporción del área de la ROI respecto a la imagen (0-1).

    Returns:
        fm (float): Medición de calidad de la región.
        rect (tuple[int, int, int, int]): Coordenadas del rectángulo ROI (x1, y1, x2, y2).
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
    fm = fm_sharpness(roi_img)

    return fm, (x1, y1, x2, y2)


def fm_sharpness_roi_video(video_path: str, roi: float = 0.05):
    """
    Calcula la métrica FM en la región central para cada frame de un video.

    Args:
        video_path (str): Ruta del video.
        roi (float): Proporción del área de la ROI respecto a la imagen (0-1).

    Returns:
        fm_values (list[float]): Valores FM por frame.
        frame_numbers (list[int]): Índices de frames.
        fm_max (float | None): Valor máximo FM en el video.
        frame_max (int | None): Índice del frame con FM máximo.
        rect (tuple[int, int, int, int]): Rectángulo de la ROI.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fm_values, frame_numbers = [], []
    frame_idx = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        fm, rect = fm_sharpness_roi(frame, roi)
        fm_values.append(fm)
        frame_numbers.append(frame_idx)
        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()

    if fm_values:
        fm_max = max(fm_values)
        frame_max = fm_values.index(fm_max)
    else:
        fm_max = None
        frame_max = None

    return fm_values, frame_numbers, fm_max, frame_max, rect


def fm_sharpness_grid(img: np.ndarray, N: int = 3, M: int = 3):
    """
    Calcula FM en una grilla NxM de cuadrados centrados en la imagen.

    Cada cuadrado tiene separación = 0.5 * lado y borde desde la imagen = 0.5 * lado.

    Args:
        img (np.ndarray): Imagen de entrada (MxN).
        N (int): Filas de la grilla.
        M (int): Columnas de la grilla.

    Returns:
        fm_matrix (np.ndarray): Matriz NxM con valores FM por bloque.
        rect_matrix (np.ndarray): Matriz NxM con coordenadas de cada bloque (x1, y1, x2, y2).
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

    fm_matrix = np.zeros((N, M), dtype=float)
    rect_matrix = np.empty((N, M), dtype=object)

    for i in range(N):
        for j in range(M):
            x1 = offset_x + j * (S + gap)
            y1 = offset_y + i * (S + gap)
            x2 = x1 + S
            y2 = y1 + S

            rect_matrix[i, j] = (int(x1), int(y1), int(x2), int(y2))
            block = img[y1:y2, x1:x2]
            fm_matrix[i, j] = fm_sharpness(block)

    return fm_matrix, rect_matrix


def fm_sharpness_grid_video(video_path: str, N: int = 3, M: int = 3):
    """
    Calcula FM usando una grilla NxM de bloques para cada frame de un video.

    Args:
        video_path (str): Ruta del video.
        N (int): Filas de la grilla.
        M (int): Columnas de la grilla.

    Returns:
        fm_values (list[float]): FM máximo de cada frame.
        frame_numbers (list[int]): Índices de frames.
        fm_max (float | None): FM máximo global del video.
        frame_max (int | None): Frame donde ocurre el FM máximo global.
        rect_max (tuple[int, int, int, int] | None): Rectángulo del FM máximo global.
        rect_matrices (list[np.ndarray]): Lista de matrices NxM con rectángulos de cada frame.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fm_values, frame_numbers = [], []
    fm_matrices, rect_matrices = [], []
    frame_idx = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        fm_matrix, rect_matrix = fm_sharpness_grid(frame, N, M)
        fm_matrices.append(fm_matrix)
        rect_matrices.append(rect_matrix)

        fm_values.append(fm_matrix.max())
        frame_numbers.append(frame_idx)
        frame_idx += 1

    video_capture.release()
    cv2.destroyAllWindows()

    if not fm_values:
        return [], [], None, None, None, rect_matrices

    fm_max = max(fm_values)
    frame_max = fm_values.index(fm_max)
    fm_matrix_target = fm_matrices[frame_max]

    i, j = np.unravel_index(np.argmax(fm_matrix_target), fm_matrix_target.shape)
    rect_max = rect_matrices[frame_max][i, j]

    return fm_values, frame_numbers, fm_max, frame_max, rect_max, rect_matrices

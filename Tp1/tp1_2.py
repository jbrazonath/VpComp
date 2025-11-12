import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Las imágenes son visualmente distintas: la Imagen 1 muestra un degradado en escala de grises, mientras que la Imagen 2 corresponde a una flor. 
Sin embargo, esta diferencia en contenido no queda reflejada en los histogramas: con 256 bins (que cubren el rango 0–255) ambos presentan 
la misma distribución de intensidades o una extremadamente similar. Es posible que la Imagen 1 se haya generado a partir de la Imagen 2 
mediante una transformación que preserva la distribución de niveles de gris.
Este resultado funciona como contraejemplo: el histograma resume la frecuencia de los niveles de gris pero ignora la disposición espacial 
de los píxeles. Por lo tanto, no identifica de forma unívoca una imagen. En tareas de clasificación o detección, el histograma puede ser un 
descriptor útil, pero conviene complementarlo con características que capturen información espacial (por ejemplo, texturas, descriptores locales 
o momentos espaciales) o aplicar verificaciones adicionales para evitar confusiones como la observada aquí.
'''

PATH_IMG_1 = "Material_TPs/TP1/img1_tp.png"
PATH_IMG_2 = "Material_TPs/TP1/img2_tp.png"

# Lectura de imagenes con CV2 flag para escala de grises
img_1 = cv2.imread(PATH_IMG_1, cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread(PATH_IMG_2, cv2.IMREAD_GRAYSCALE)

# --- Histogramas ---
# Usamos 256 bins para cubrir el rango dinámico completo de 8 bits: niveles 0..255
hist_1, bins_1 = np.histogram(img_1.ravel(), 255)
hist_2, bins_2 = np.histogram(img_2.ravel(), 255)

# --- Visualización ---
fig_1 = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(221)
ax1.imshow(img_1, cmap='gray', vmin=0, vmax=255)
ax1.set_title("Imagen 1")

ax2 = plt.subplot(222)
ax2.imshow(img_2, cmap='gray', vmin=0, vmax=255)
ax2.set_title("Imagen 2")

ax3 = plt.subplot(223)
ax3.plot(hist_1)
ax3.set_title("Histograma Imagen 1")

ax4 = plt.subplot(224)
ax4.plot(hist_2)
ax4.set_title("Histograma Imagen 2")

plt.show()
plt.savefig("results/histogramas_tp1.png")


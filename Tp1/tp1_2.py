import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH_IMG_1 = "Material_TPs/TP1/img1_tp.png"
PATH_IMG_2 = "Material_TPs/TP1/img2_tp.png"

img_1 = cv2.imread(PATH_IMG_1, cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread(PATH_IMG_2, cv2.IMREAD_GRAYSCALE)

# --- Histogramas ---
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
plt.savefig("results/histogramas.png")


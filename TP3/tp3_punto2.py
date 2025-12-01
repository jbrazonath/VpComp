import cv2
import numpy as np
import os

# --- CONFIGURACIÓN ---
path_img = 'TP3/images/coca_multi.png'
path_template = 'TP3/template/pattern.png'

# 1. CARGAR IMÁGENES
if not os.path.exists(path_img) or not os.path.exists(path_template):
    print("¡Error! Revisa las rutas de los archivos.")
    exit()

img_rgb = cv2.imread(path_img)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(path_template, 0)

# TRUCO CLAVE: Usar el negativo (letras blancas sobre fondo oscuro)
template_invertido = cv2.bitwise_not(template)
h_orig, w_orig = template.shape

print("--- EJERCICIO 2: DETECCIÓN MÚLTIPLE ---")
print("Procesando 'coca_multi.png' con Template Matching Multi-Escala...")

detecciones = []

# 2. ESCANEO DE TAMAÑOS
for escala in np.linspace(0.12, 0.65, 100):
    w_new = int(w_orig * escala)
    h_new = int(h_orig * escala)
    
    if w_new < 10 or h_new < 10: continue
    
    t_inv_escalado = cv2.resize(template_invertido, (w_new, h_new))
    res = cv2.matchTemplate(img_gray, t_inv_escalado, cv2.TM_CCOEFF_NORMED)
    
    loc = np.where(res >= 0.50)
    
    for pt in zip(*loc[::-1]):
        detecciones.append([int(pt[0]), int(pt[1]), int(w_new), int(h_new)])

# 3. FILTRADO
rects_agrupados, pesos = cv2.groupRectangles(detecciones, groupThreshold=3, eps=0.3)

print(f"-> Se detectaron {len(rects_agrupados)} botellas válidas.")

# 4. DIBUJAR
for (x, y, w, h) in rects_agrupados:
    ratio = w / float(h)
    if 1.5 < ratio < 4.0: 
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # --- AQUÍ ESTABA EL ERROR (Corregido a MAYÚSCULAS) ---
        cv2.putText(img_rgb, "Coca-Cola", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imshow('Ejercicio 2 - Detecciones Multiples', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
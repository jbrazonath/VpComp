import cv2
import numpy as np
import os

# --- CONFIGURACIÓN ---
carpeta_imagenes = 'TP3/images/'
path_template = 'TP3/template/pattern.png'

template_normal = cv2.imread(path_template, 0)
if template_normal is None:
    print("Error: No hay template")
    exit()

template_invertido = cv2.bitwise_not(template_normal)
h_orig, w_orig = template_normal.shape

# Configurar ORB
orb = cv2.ORB_create(nfeatures=15000, scaleFactor=1.1, nlevels=20, edgeThreshold=5, patchSize=31)
kp1_norm, des1_norm = orb.detectAndCompute(template_normal, None)
kp1_inv, des1_inv = orb.detectAndCompute(template_invertido, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

print(f"--- INICIANDO SISTEMA CON FILTRO DE CONSENSO ---")
print("Presiona ESPACIO para avanzar.\n")

for nombre_archivo in os.listdir(carpeta_imagenes):
    if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path_img_actual = os.path.join(carpeta_imagenes, nombre_archivo)
    img_rgb = cv2.imread(path_img_actual)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_final = img_rgb.copy()

    # =========================================================================
    # ESTRATEGIA A: COCA MULTI (Filtrado por Consenso)
    # =========================================================================
    if "multi" in nombre_archivo.lower():
        print(f"[{nombre_archivo}] -> Escaneando (Alta Densidad)...")
        detecciones = []
        
        # 1. ESCANEO ULTRA DENSO (100 pasos)
        # Probamos muchísimos tamaños. Las botellas reales darán positivo muchas veces seguidas.
        for escala in np.linspace(0.12, 0.65, 100):
            w_new = int(w_orig * escala)
            h_new = int(h_orig * escala)
            if w_new < 10: continue
            
            t_inv_escalado = cv2.resize(template_invertido, (w_new, h_new))
            
            # Match
            res = cv2.matchTemplate(img_gray, t_inv_escalado, cv2.TM_CCOEFF_NORMED)
            
            # Umbral Moderado (0.50): Ni muy bajo (ruido) ni muy alto (perdemos botellas)
            loc = np.where(res >= 0.50)
            
            for pt in zip(*loc[::-1]):
                detecciones.append([int(pt[0]), int(pt[1]), int(w_new), int(h_new)])

        if len(detecciones) > 0:
            # 2. LA CLAVE: groupThreshold = 4
            # Significa: "Solo dibuja si al menos 5 rectángulos cayeron en el mismo lugar"
            # El ruido suele ser aleatorio y no se agrupa tanto. Las botellas son firmes.
            rects_agrupados, pesos = cv2.groupRectangles(detecciones, groupThreshold=4, eps=0.3)
            
            if len(rects_agrupados) == 0:
                # Si fuimos muy estrictos, bajamos la exigencia un poco (Plan B)
                print("   >>> Reintentando con menor exigencia de grupo...")
                rects_agrupados, pesos = cv2.groupRectangles(detecciones, groupThreshold=2, eps=0.3)

            count = 0
            for (x, y, w, h) in rects_agrupados:
                # Filtro final de geometría: El logo es rectangular (más ancho que alto)
                # Si detectó algo cuadrado o vertical (ruido raro), lo ignoramos
                ratio = w / float(h)
                if 1.5 < ratio < 4.0: # El logo de Coca-Cola es alargado
                    cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    count += 1
            print(f"   >>> ¡DIBUJANDO {count} BOTELLAS LIMPIAS!")
        else:
            print("   >>> No se encontraron coincidencias.")

    # =========================================================================
    # ESTRATEGIA B: ORB DUAL (Resto de imágenes)
    # =========================================================================
    else:
        kp2, des2 = orb.detectAndCompute(img_gray, None)
        if des2 is not None:
            def buscar_coincidencias(des_template, des_img):
                if des_template is None or des_img is None: return []
                matches = bf.knnMatch(des_template, des_img, k=2)
                good = []
                try:
                    for m, n in matches:
                        if m.distance < 0.92 * n.distance:
                            good.append(m)
                except ValueError: pass
                return good

            good_norm = buscar_coincidencias(des1_norm, des2)
            good_inv = buscar_coincidencias(des1_inv, des2)

            if len(good_norm) >= len(good_inv):
                good_matches = good_norm
                kp_template = kp1_norm
                tipo = "Normal"
            else:
                good_matches = good_inv
                kp_template = kp1_inv
                tipo = "Invertido"

            print(f"[{nombre_archivo}] -> ORB ({tipo}) | Puntos: {len(good_matches)}")

            if len(good_matches) >= 5:
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)

                if M is not None:
                    pts = np.float32([[0, 0], [0, h_orig - 1], [w_orig - 1, h_orig - 1], [w_orig - 1, 0]]).reshape(-1, 1, 2)
                    try:
                        dst = cv2.perspectiveTransform(pts, M)
                        dst_pts_int = np.int32(dst)
                        if cv2.contourArea(dst_pts_int) > 300:
                            overlay = img_final.copy()
                            cv2.fillPoly(overlay, [dst_pts_int], (0, 255, 0))
                            cv2.addWeighted(overlay, 0.3, img_final, 0.7, 0, img_final)
                            cv2.polylines(img_final, [dst_pts_int], True, (0, 255, 0), 3, cv2.LINE_AA)
                    except: pass

    cv2.imshow('TP3 Final - Sin Falsos Positivos', img_final)
    cv2.waitKey(0)

cv2.destroyAllWindows()
# =============================================================================
# imagenes_a_csv.py
# Lee imágenes binarizadas y las convierte a un CSV con columnas pixel_1..pixel_16384.
#
# Requisitos:
#   pip install opencv-python numpy
#
# Estructura de carpetas esperada:
#   imagenes_entrenamiento/
#       1.1.png, 1.2.png ...   <- negativos (sin arroz)
#       2.1.png, 2.2.png ...   <- positivos (con arroz)
#   imagenes_prueba/
#       negativa_1.png ...     <- negativos
#       positiva_1.png ...     <- positivos
#
# Uso:
#   python imagenes_a_csv.py
# =============================================================================

import os
import re
import csv
import cv2
import numpy as np

# =============================================================================
# CONFIGURACIÓN — ajustá las rutas si tus carpetas tienen otro nombre
# =============================================================================

DIR_IMAGENES_ENTRENAMIENTO = "imagenes_entrenamiento"
DIR_IMAGENES_PRUEBA        = "imagenes_prueba"

RUTA_CSV_ENTRENAMIENTO = "dataset_entrenamiento.csv"
RUTA_CSV_PRUEBA        = "dataset_prueba.csv"

ANCHO, ALTO = 128, 128
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# =============================================================================
# FUNCIONES
# =============================================================================

def crear_estructura_carpetas():
    for ruta in [DIR_IMAGENES_ENTRENAMIENTO, DIR_IMAGENES_PRUEBA]:
        if not os.path.exists(ruta):
            os.makedirs(ruta)
            print(f"Carpeta creada: {ruta}")


def leer_imagen_binaria_como_vector(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise ValueError(f"No se pudo leer: {ruta_imagen}")
    imagen = cv2.resize(imagen, (ANCHO, ALTO), interpolation=cv2.INTER_NEAREST)
    imagen_binaria = np.where(imagen > 127, 1, 0).astype(np.uint8)
    return imagen_binaria.flatten().tolist()


def obtener_imagenes_validas(directorio):
    return [
        os.path.join(directorio, f)
        for f in os.listdir(directorio)
        if os.path.isfile(os.path.join(directorio, f))
        and f.lower().endswith(EXTENSIONES_VALIDAS)
    ]


def extraer_numero_entrenamiento(ruta_imagen):
    nombre = os.path.splitext(os.path.basename(ruta_imagen))[0]
    coincidencia = re.match(r"^[12]\.(\d+)$", nombre)
    return int(coincidencia.group(1)) if coincidencia else 999999


def extraer_numero_prueba(ruta_imagen):
    nombre = os.path.splitext(os.path.basename(ruta_imagen))[0].lower()
    coincidencia = re.search(r"(\d+)$", nombre)
    return int(coincidencia.group(1)) if coincidencia else 999999


def clasificar_imagen_entrenamiento(ruta_imagen):
    nombre = os.path.basename(ruta_imagen)
    if nombre.startswith("1."):
        return 0   # negativo (sin arroz)
    if nombre.startswith("2."):
        return 1   # positivo (con arroz)
    raise ValueError(f"Nombre inválido en entrenamiento: {nombre}")


def clasificar_imagen_prueba(ruta_imagen):
    nombre = os.path.basename(ruta_imagen).lower()
    if nombre.startswith(("negativa", "negativo")):
        return 0
    if nombre.startswith(("positiva", "positivo")):
        return 1
    raise ValueError(f"Nombre inválido en prueba: {nombre}")


def generar_csv(directorio_imgs, ruta_csv, func_clasificar, func_extraer, titulo):
    print(f"\n{'=' * 45}")
    print(titulo)
    print("=" * 45)

    imgs = obtener_imagenes_validas(directorio_imgs)
    imgs.sort(key=func_extraer)

    if not imgs:
        print(f"Aviso: No hay imágenes en '{directorio_imgs}'.")
        return

    n_pixeles  = ANCHO * ALTO
    encabezado = [f"pixel_{i + 1}" for i in range(n_pixeles)] + ["etiqueta"]

    with open(ruta_csv, mode="w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(encabezado)

        for i, ruta in enumerate(imgs, 1):
            etiqueta = func_clasificar(ruta)
            vector   = leer_imagen_binaria_como_vector(ruta)
            escritor.writerow(vector + [etiqueta])
            print(f"  Procesado {str(i).zfill(2)}: {os.path.basename(ruta)} -> etiqueta={etiqueta}")

    print(f"\n  CSV guardado : {ruta_csv}")
    print(f"  Total        : {len(imgs)} imágenes")

# =============================================================================
# EJECUCIÓN
# =============================================================================

def main():
    crear_estructura_carpetas()

    # Verificar que existan imágenes antes de procesar
    faltan = []
    for carpeta in [DIR_IMAGENES_ENTRENAMIENTO, DIR_IMAGENES_PRUEBA]:
        if not obtener_imagenes_validas(carpeta):
            faltan.append(carpeta)

    if faltan:
        print("\nNo se encontraron imágenes en las siguientes carpetas:")
        for c in faltan:
            print(f"  {c}")
        print("\nAsegurate de tener imágenes con los nombres correctos:")
        print("  Entrenamiento: 1.1.png (negativo), 2.1.png (positivo)")
        print("  Prueba       : negativa_1.png, positiva_1.png")
        return

    generar_csv(
        DIR_IMAGENES_ENTRENAMIENTO,
        RUTA_CSV_ENTRENAMIENTO,
        clasificar_imagen_entrenamiento,
        extraer_numero_entrenamiento,
        "GENERANDO CSV — ENTRENAMIENTO"
    )

    generar_csv(
        DIR_IMAGENES_PRUEBA,
        RUTA_CSV_PRUEBA,
        clasificar_imagen_prueba,
        extraer_numero_prueba,
        "GENERANDO CSV — PRUEBA"
    )

    print("\nProceso finalizado.")
    print(f"  {RUTA_CSV_ENTRENAMIENTO}")
    print(f"  {RUTA_CSV_PRUEBA}")


if __name__ == "__main__":
    main()

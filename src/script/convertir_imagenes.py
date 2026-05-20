# =============================================================================
# convertir_imagenes.py
# Convierte imágenes HEIC (y otros formatos) a PNG 128x128 en blanco y negro.
#
# Requisitos:
#   pip install pillow pillow-heif matplotlib
#
# Uso:
#   Colocá las imágenes en la carpeta de entrada (INPUT_DIR) y ejecutá:
#   python convertir_imagenes.py
# =============================================================================

import io
import shutil
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener
import matplotlib.pyplot as plt

register_heif_opener()

# =============================================================================
# CONFIGURACIÓN — ajustá estas rutas según tus carpetas
# =============================================================================

INPUT_DIR  = Path("imagenes_originales")   # carpeta con las imágenes de entrada
OUTPUT_DIR = Path("converted")             # carpeta donde se guardan los PNG convertidos

EXTENSIONES_VALIDAS = (".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# =============================================================================
# CONVERSIÓN
# =============================================================================

def convertir_imagen(ruta: Path, output_dir: Path) -> Path | None:
    try:
        img = Image.open(ruta)
        img = img.convert("L")                          # escala de grises
        img = img.resize((128, 128), Image.LANCZOS)     # 128x128

        out_path = output_dir / (ruta.stem + ".png")
        img.save(out_path, format="PNG")
        print(f"  {ruta.name} -> {out_path.name}")
        return out_path

    except Exception as e:
        print(f"  ERROR: {ruta.name} - {e}")
        return None


def main():
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True)
        print(f"Carpeta de entrada creada: {INPUT_DIR}")
        print("Colocá tus imágenes ahí y volvé a ejecutar el script.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archivos = [f for f in INPUT_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in EXTENSIONES_VALIDAS]

    if not archivos:
        print(f"No se encontraron imágenes en '{INPUT_DIR}'.")
        print(f"Extensiones aceptadas: {EXTENSIONES_VALIDAS}")
        return

    print(f"\nConvirtiendo {len(archivos)} imagen(es)...\n")
    convertidas = [convertir_imagen(f, OUTPUT_DIR) for f in archivos]
    convertidas = [p for p in convertidas if p is not None]

    print(f"\nListo: {len(convertidas)} imagen(es) convertida(s) en '{OUTPUT_DIR}'.")

    # Vista previa de las imágenes convertidas
    if convertidas:
        cols = min(len(convertidas), 4)
        rows = (len(convertidas) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = [axes] if len(convertidas) == 1 else axes.flatten()

        for ax, path in zip(axes, convertidas):
            ax.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
            ax.set_title(path.name, fontsize=8)
            ax.axis("off")

        for ax in axes[len(convertidas):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("vista_previa_convertidas.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Vista previa guardada: vista_previa_convertidas.png")

        # Comprimir resultados en ZIP
        zip_nombre = "converted_images"
        shutil.make_archive(zip_nombre, "zip", OUTPUT_DIR)
        print(f"ZIP creado: {zip_nombre}.zip")


if __name__ == "__main__":
    main()

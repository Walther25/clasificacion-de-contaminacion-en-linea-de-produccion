# DATASET.md — Clasificación de Contaminaciones en Línea de Producción

## Resumen

| Característica | Valor |
|---|---|
| Total de imágenes | 120 |
| Clases | Positivo (con arroz) / Negativo (sin arroz) |
| Balance | 60 / 60 — 50% por clase |
| Resolución | 128 × 128 px |
| Tipo de píxel | Binario (0 / 1) |
| Formato final | CSV — `pixel_1` … `pixel_16384` + `etiqueta` |
| Archivos fuente | `data/raw/` (HEIC) → `data/procesed/` (PNG) → `data/dataset.csv` |

---

## Recolección

Las imágenes se capturaron con un iPhone (formato HEIC) sobre superficies planas en interiores, simulando una línea de producción simplificada:

- **Clase positiva (etiqueta = 1):** superficie con granos de arroz dispersos.
- **Clase negativa (etiqueta = 0):** superficie con clips de oficina, sin arroz.

No se usó ningún equipo especializado ni condiciones controladas de iluminación. Las fotos varían en distancia, ángulo y temperatura de color según las condiciones del momento.

## Preprocesamiento

1. **Conversión** — de HEIC a PNG en escala de grises (`PIL`, modo `"L"`), redimensionado a 128×128 con `Image.LANCZOS`.
2. **Binarización** — umbral fijo: píxel > 127 → 1, resto → 0 (`OpenCV`, `INTER_NEAREST`).
3. **Vectorización** — la matriz 128×128 se aplana a un vector de 16 384 valores por fila.
4. **Etiquetado** — deducido automáticamente del nombre del archivo (ver convención abajo).

### Convención de nombres

| Prefijo | Clase | Etiqueta |
|---|---|---|
| `1.N.png` | Negativo | 0 |
| `2.N.png` | Positivo | 1 |

No hubo etiquetado manual posterior; la clase es inherente al nombre del archivo asignado durante la captura.

---

## Limitaciones

**Umbral de binarización fijo**
El valor 127 es global. Imágenes con sobreexposición o subexposición pueden producir vectores donde los objetos de interés quedan completamente blancos o negros, perdiendo estructura visual.

**Variabilidad no controlada**
Las fotos se tomaron en condiciones distintas de iluminación, distancia y ángulo. Esto introduce ruido sistemático que el modelo puede aprender como rasgo de clase en lugar del objeto real.

**Objetos de laboratorio**
Las clases no representan una línea de producción real. El modelo no generalizaría a otros contaminantes ni a otros objetos negativos sin ser reentrenado.

**Tamaño reducido**
Con 120 imágenes, el dataset es insuficiente para técnicas que requieren grandes volúmenes (redes neuronales, augmentation extensiva). Los resultados son sensibles a la semilla de aleatoriedad.

**Sin diversidad de fondo**
Todas las imágenes comparten fondos similares (superficies claras de escritorio). Un modelo entrenado aquí fallará en fondos con texturas, colores o patrones distintos.

**Sin aumentación de datos**
No se aplicaron rotaciones, volteos, ruido ni variaciones de brillo. El dataset refleja únicamente las condiciones reales de captura.

---

## Reproducibilidad

Para generar el dataset desde cero:

```bash
# Paso 1 — convertir fotos a PNG 128x128 B&N
# Colocar fotos en data/raw/ y ejecutar:
python src/script/convertir_imagenes.py

# Paso 2 — vectorizar y generar CSV
python src/script/imagenes_a_csv.py
```

Ver [README.md](README.md) para la convención de nombres de archivos y los requisitos de instalación.

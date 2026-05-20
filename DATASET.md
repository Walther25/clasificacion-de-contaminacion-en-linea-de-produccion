# Dataset — Clasificación de Contaminaciones en Línea de Producción

## Descripción general

El dataset consiste en imágenes binarizadas de objetos sobre una superficie, capturadas manualmente por los integrantes del equipo. Cada imagen representa una escena de una posible línea de producción con dos posibles condiciones:

- **Positivo (etiqueta = 1):** imagen con granos de arroz presentes (contaminación)
- **Negativo (etiqueta = 0):** imagen sin granos de arroz (sin contaminación, solo clips)

---

## Composición

| Característica | Valor |
|---|---|
| Total de imágenes | 120 |
| Imágenes positivas | 60 (con arroz) |
| Imágenes negativas | 60 (sin arroz) |
| Balance de clases | 50% / 50% |
| Resolución | 128 × 128 píxeles |
| Formato almacenado | CSV (una fila por imagen) |
| Columnas por imagen | 16 384 (`pixel_1` … `pixel_16384`) + `etiqueta` |
| Tipo de píxel | Binario (0 = negro, 1 = blanco) |
| Integrantes | 4 (30 imágenes por persona) |

---

## Proceso de recolección

### 1. Captura de imágenes

Cada integrante capturó **30 imágenes** con la cámara de su celular:
- 15 imágenes positivas: superficie con granos de arroz dispersos
- 15 imágenes negativas: superficie con clips (objeto de referencia, sin arroz)

Las fotos se tomaron en formato HEIC (iPhone) o JPG/PNG dependiendo del dispositivo.

### 2. Conversión a PNG 128×128 en blanco y negro

Usando el script `convertir_imagenes.py`:
- Conversión de HEIC a PNG con la librería `pillow-heif`
- Redimensionado a 128 × 128 píxeles (`Image.LANCZOS`)
- Conversión a escala de grises (`convert("L")`)

### 3. Binarización y extracción de píxeles

Usando el script `imagenes_a_csv.py`:
- Lectura de cada PNG con OpenCV en escala de grises
- Umbral de binarización: píxeles > 127 → 1, resto → 0
- Aplanado de la matriz 128×128 a un vector de 16 384 valores
- Escritura en CSV con encabezado `pixel_1, ..., pixel_16384, etiqueta`

### 4. Convención de nombres de archivos

| Conjunto | Prefijo | Etiqueta |
|---|---|---|
| Entrenamiento | `1.N.png` | 0 (negativo) |
| Entrenamiento | `2.N.png` | 1 (positivo) |
| Prueba | `negativa_N.png` | 0 (negativo) |
| Prueba | `positiva_N.png` | 1 (positivo) |

### 5. Combinación del dataset final

Los 4 CSVs individuales (uno por integrante) se combinaron en un único `dataset.csv` usando pandas, normalizando los encabezados al formato estándar `pixel_1 … pixel_16384, etiqueta`.

---

## Limitaciones

### Tamaño reducido
Con solo 120 imágenes el dataset es pequeño para problemas de visión por computadora. Esto hace que los resultados sean sensibles a la partición de datos, por lo que se usó evaluación repetida de 20 iteraciones para obtener métricas más confiables.

### Variabilidad entre capturas
Cada integrante tomó sus fotos en condiciones distintas de iluminación, distancia, ángulo y fondo. Esto introduce ruido que puede afectar la generalización de los modelos.

### Binarización con umbral fijo
El umbral de binarización (127) es fijo para todas las imágenes. Imágenes con iluminación muy distinta pueden resultar en vectores de píxeles poco representativos si el contraste no es suficiente.

### Objetos limitados
Las clases no representan una línea de producción real: los negativos son clips de oficina y los positivos son granos de arroz. El modelo entrenado no generalizaría a otros tipos de contaminantes sin ser reentrenado.

### Resolución reducida
Bajar a 128×128 píxeles comprime la información espacial. Objetos pequeños o con poca diferencia de contraste respecto al fondo pueden perderse en la binarización.

### Sin aumentación de datos
No se aplicaron técnicas de data augmentation (rotaciones, volteos, ruido). El dataset refleja únicamente las condiciones reales de captura de cada integrante.

---

## Reproducibilidad

Para reproducir el dataset desde cero:

1. Capturá imágenes siguiendo la convención de nombres descrita arriba.
2. Corré `convertir_imagenes.py` para convertir a PNG 128×128 B&N.
3. Corré `imagenes_a_csv.py` para generar el CSV con píxeles.
4. Combiná los CSVs individuales con pandas o usá el script de combinación disponible en el repositorio.

Ver el `README.md` para instrucciones detalladas de cada paso.

---

## Integrantes del equipo

| Nombre | Carné | Dataset individual |
|---|---|---|
| Walther Barrantes | C31037 | `dataset_walther.csv` |
| Kenneth | — | `dataset_kenneth.csv` |
| Marlon | — | `dataset_marlon.csv` |
| Sebastian | — | `dataset_sebastian.csv` |
| Danna | — | `dataset_danna.csv` |

> **Nota:** el dataset combinado final (`dataset.csv`) usado en los experimentos reportados en `resultados.txt` corresponde a los archivos de Marlon, Sebastian, Walther y Danna (120 imágenes).

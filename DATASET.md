# Dataset — Clasificación de Contaminaciones en Línea de Producción

## Descripción general

El dataset consiste en imágenes binarizadas de objetos sobre una superficie, capturadas manualmente con la cámara de un celular. Cada imagen representa una escena de una posible línea de producción con dos condiciones posibles:

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

---

## Proceso de recolección

### 1. Captura de imágenes

Las imágenes se tomaron con la cámara de un celular en formato HEIC, JPG o PNG:
- Imágenes positivas: superficie con granos de arroz dispersos
- Imágenes negativas: superficie con clips (objeto de referencia, sin arroz)

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

### 5. Formato del CSV final

El archivo `dataset.csv` tiene el encabezado estándar `pixel_1 … pixel_16384, etiqueta`. Si subís varios CSVs, el notebook los combina automáticamente en un único dataset.

---

## Limitaciones

### Tamaño reducido
Con 120 imágenes el dataset es pequeño para problemas de visión por computadora. Los resultados son sensibles a la partición de datos, por lo que se usó evaluación repetida de 20 iteraciones para obtener métricas más confiables.

### Condiciones de captura variables
Las fotos se tomaron en condiciones distintas de iluminación, distancia, ángulo y fondo. Esto introduce ruido que puede afectar la generalización de los modelos.

### Binarización con umbral fijo
El umbral de binarización (127) es fijo para todas las imágenes. Con iluminación muy distinta entre capturas, el contraste puede no ser suficiente y los vectores de píxeles resultantes pueden no ser representativos.

### Objetos de laboratorio
Las clases no representan una línea de producción real: los negativos son clips de oficina y los positivos son granos de arroz. El modelo no generalizaría a otros tipos de contaminantes sin ser reentrenado con nuevos datos.

### Resolución reducida
Reducir a 128×128 píxeles comprime la información espacial. Objetos pequeños o con poco contraste respecto al fondo pueden perderse en la binarización.

### Sin aumentación de datos
No se aplicaron técnicas de data augmentation (rotaciones, volteos, ruido). El dataset refleja únicamente las condiciones reales de captura.

---

## Reproducibilidad

Para generar tu propio dataset desde cero:

1. Tomá fotos siguiendo la convención de nombres descrita arriba.
2. Corré `convertir_imagenes.py` para convertir a PNG 128×128 B&N.
3. Corré `imagenes_a_csv.py` para generar el CSV con píxeles.
4. Subí el CSV al notebook o pasalo como argumento al script `clasificacion.py`.

Ver el `README.md` para instrucciones detalladas de cada paso.

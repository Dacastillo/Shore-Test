# Carpeta de datos — `data/`

Contiene los datos utilizados y generados durante el análisis de línea de costa
con CoastSat.

## Archivos principales

### Archivos MATLAB (.mat)

Estos archivos contienen variables y resultados intermedios del procesamiento:

| Archivo | Descripción |
|---------|-------------|
| `Edit_Shoreline_Python.mat` | Coordenadas de la línea de costa editada |
| `AxesImage4Handles.mat` | Datos de imagen para visualización |
| `Georectify_Python.mat` | Parámetros de georectificación |
| `MapShorelinePython.mat` | Datos de mapeo de línea de costa |
| `build___rect___products___Python.mat` | Productos rectificados |
| `GlobsOutput.mat` | Resultados de operaciones morfológicas |
| `load_Image_Python.mat` | Imágenes cargadas para procesamiento |
| `RectifyImagePython.mat` | Parámetros de rectificación de imagen |
| `UV_computed.mat` | Coordenadas UV computadas |

### Archivos de CoastSnap

| Archivo | Descripción |
|---------|-------------|
| `CoastSnapDB.xlsx` | Base de datos de imágenes CoastSnap |

### Carpeta pettenzuid

Contiene imágenes de ejemplo del sitio Petten aan Zee (Países Bajos):

- `Raw/2022/`: Imágenes originales sin procesar
- `Processed/2022/`: Imágenes procesadas y rectificadas

## ⚠️ Nota sobre imágenes satelitales

Las imágenes satelitales descargadas desde Google Earth Engine (subcarpetas
`jpg_files/`, `ms/`, `pan/`, `swir/`) **no están incluidas en este repositorio**
debido a su tamaño. Para regenerarlas, ejecutar las celdas de descarga en el
notebook principal `Shore-Test-integrado.ipynb`.

## Sistema de referencia de coordenadas

Las coordenadas están expresadas en el sistema de proyección especificado en
`settings['output_epsg']` del notebook principal. Consultar `doc/methods.md`
para más detalles.

## Cómo regenerar los datos

```bash
# Activar entorno y abrir notebook
conda activate coastsat
jupyter notebook Shore-Test-integrado.ipynb

# Ejecutar las celdas de descarga y procesamiento
```

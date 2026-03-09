# Metodología — Parámetros de CoastSat utilizados

## Versión de CoastSat

Este repositorio utiliza el toolkit CoastSat para la extracción de líneas de costa.
La implementación se basa en los módulos principales de CoastSat:

- `SDS_download`: Descarga de imágenes desde Google Earth Engine
- `SDS_preprocess`: Preprocesamiento y corrección de imágenes
- `SDS_shoreline`: Extracción de la línea de costa
- `SDS_tools`: Utilidades generales
- `SDS_classify`: Clasificación de píxeles

## Parámetros de descarga de imágenes

La configuración de descarga se define en el notebook principal `Shore-Test-integrado.ipynb`:

```python
inputs = {
    'sitename':   '[nombre del sitio]',
    'polygon':    [coordenadas del KML],
    'dates':      [fecha_inicio, fecha_fin],
    'sat_list':   ['L5', 'L7', 'L8', 'L9', 'S2'],  # Satélites a utilizar
    'filepath':   'data/',
}
```

## Parámetros de procesamiento

Los parámetros principales de configuración incluyen:

```python
settings = {
    'cloud_thresh':      0.1,       # Umbral máximo de cobertura de nubes (0-1)
    'dist_clouds':       500,       # Distancia mínima a nubes (metros)
    'output_epsg':       [EPSG local],  # Sistema de coordenadas de salida
    'check_detection':   True,      # Verificación visual de detección
    'adjust_detection':  False,     # Ajuste manual de parámetros
    'save_figure':       True,      # Guardar figuras de resultado
    'min_beach_area':    100,       # Área mínima de playa detectada (m²)
    'min_length_sl':     50,        # Longitud mínima de línea de costa (m)
    'sand_color':        'default', # Esquema de color de arena
    'pan_off':           False,     # Pansharpening activado/desactivado
    's2cloudless_prob':  60,        # Probabilidad de nubes para Sentinel-2
}
```

## Clasificadores disponibles

La carpeta `models/` contiene los siguientes clasificadores:

| Archivo | Tipo | Satélite | Descripción |
|---------|------|----------|-------------|
| `NN_4classes_Landsat.pkl` | MLP | Landsat | Clasificador por defecto |
| `NN_4classes_Landsat_new.pkl` | MLP | Landsat | Versión actualizada |
| `NN_4classes_Landsat_dark.pkl` | MLP | Landsat | Para arena oscura |
| `NN_4classes_Landsat_bright.pkl` | MLP | Landsat | Para arena clara |
| `NN_4classes_S2.pkl` | MLP | Sentinel-2 | Clasificador S2 |
| `NN_4classes_S2_new.pkl` | MLP | Sentinel-2 | Versión actualizada S2 |
| `CoastSat_training_set_L8.pkl` | Dataset | Landsat 8 | Conjunto de entrenamiento |
| `CoastSat_training_set_S2.pkl` | Dataset | Sentinel-2 | Conjunto de entrenamiento |

### Clases de clasificación

El clasificador identifica 4 categorías de píxeles:

1. **Arena:** Playas y dunas
2. **Agua:** Océano y cuerpos de agua
3. **Espuma/rompiente:** Zona de surf y espuma
4. **Otras superficies terrestres:** Vegetación, rocas, infraestructura

## Transectos para análisis temporal

Los transectos se definen perpendicularmente a la línea de costa para analizar
la variación temporal de la posición de la playa. La configuración típica incluye:

- **Espaciado:** 20-50 m entre transectos
- **Longitud:** Suficiente para cubrir la zona de variación estacional
- **Orientación:** Perpendicular a la tendencia general de la costa

## Corrección de marea

La corrección de marea puede aplicarse para referir todas las mediciones a un
nivel de referencia común. Consultar el notebook principal para ver si se
aplicó corrección de marea en este análisis.

## Flujo de trabajo

1. **Definición de ROI:** Cargar polígono desde archivo KML
2. **Descarga:** Obtener imágenes desde Google Earth Engine
3. **Preprocesamiento:** Corrección atmosférica y pansharpening
4. **Clasificación:** Aplicar clasificador neuronal
5. **Detección de costa:** Extraer línea de costa sub-píxel
6. **Validación:** Verificación visual y ajuste si es necesario
7. **Análisis:** Generar series temporales por transecto

## Referencias

- Vos, K., et al. (2019). CoastSat: A Google Earth Engine-enabled Python toolkit
  to extract shorelines from publicly available satellite imagery.
  *Environmental Modelling & Software*, 122, 104528.

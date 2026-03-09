# Shore-Test — Análisis de línea de costa con imágenes satelitales

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![CoastSat](https://img.shields.io/badge/Basado_en-CoastSat-green)](https://github.com/kvos/CoastSat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Aplicación del toolkit [CoastSat](https://github.com/kvos/CoastSat) para el rastreo
de la posición de la línea de costa a partir de imágenes satelitales Landsat y Sentinel-2.

## Descripción del proyecto

Este repositorio implementa un flujo de trabajo completo para:

1. **Descarga de imágenes satelitales** desde Google Earth Engine (GEE) para la
   región de interés definida en `kml/`
2. **Preprocesamiento**: corrección atmosférica, enmascaramiento de nubes,
   pansharpening para Landsat 7/8/9
3. **Clasificación pixel a pixel** en 4 categorías: arena, agua, espuma/romper
   de olas y otras superficies terrestres, usando el clasificador en `models/`
4. **Extracción sub-píxel de la línea de costa** mediante segmentación de borde
   (algoritmo MNDWI/NDWI + Otsu threshold)
5. **Análisis temporal** de la variación de la posición de la costa en transectos
   perpendiculares

## Sitio estudiado

Este repositorio contiene datos y análisis para sitios costeros en Australia:

- **Ubicación:** Sawtell, Newcastle, Byron Bay (Nueva Gales del Sur, Australia)
- **Coordenadas:** Ver archivos KML en `kml/` para los polígonos exactos
- **Satélites:** Landsat y Sentinel-2
- **Período analizado:** Ver notebook principal para detalles

## Estructura del repositorio

```
Shore-Test/
├── Shore-Test-integrado.ipynb   # Notebook principal: flujo completo de análisis
├── example_jupyter.ipynb        # Notebook de referencia de CoastSat
├── funciones_shore.py           # Funciones auxiliares propias
├── data/                        # Imágenes satelitales y resultados
├── kml/                         # Región de interés (ROI) para Google Earth Engine
├── models/                      # Clasificadores entrenados (.pkl)
└── doc/                         # Documentación técnica y metodológica
```

## Instalación

```bash
# 1. Clonar el repositorio
git clone git@github.com:Dacastillo/Shore-Test.git
cd Shore-Test

# 2. Crear entorno conda (recomendado sobre pip para dependencias geoespaciales)
conda create -n coastsat python=3.10
conda activate coastsat

# 3. Instalar dependencias
conda install -c conda-forge geopandas
conda install -c conda-forge earthengine-api scikit-image matplotlib astropy notebook
pip install -r requirements.txt

# 4. Autenticar Google Earth Engine
earthengine authenticate
```

## Uso

### Análisis completo (notebook integrado)

```bash
conda activate coastsat
jupyter notebook Shore-Test-integrado.ipynb
```

Ejecutar las celdas en orden. Ver `doc/methods.md` para una descripción detallada
de cada paso y los parámetros configurables.

### Reentrenar el clasificador para este sitio

Ver `doc/train_new_classifier.md` para instrucciones de cómo añadir nuevas imágenes
etiquetadas y reentrenar el modelo en `models/`.

## Base teórica y referencia de CoastSat

Este trabajo usa el algoritmo de CoastSat:

> Vos, K., Splinter, K. D., Harley, M. D., Simmons, J. A., & Turner, I. L. (2019).
> CoastSat: A Google Earth Engine-enabled Python toolkit to extract shorelines from
> publicly available satellite imagery.
> *Environmental Modelling & Software*, 122, 104528.
> https://doi.org/10.1016/j.envsoft.2019.104528

## Autor

Daniel Castillo-Castro · [dacastillo.github.io](http://dacastillo.github.io) · Santiago, Chile

## Licencia

MIT © Daniel Castillo-Castro

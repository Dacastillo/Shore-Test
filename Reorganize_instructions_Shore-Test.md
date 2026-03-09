# Instrucciones de reorganización y documentación — `Dacastillo/Shore-Test`

> **Destinatario:** Gestor de código autónomo (Claude Code, Copilot Workspace o similar)
> **Repositorio objetivo:** https://github.com/Dacastillo/Shore-Test
> **Nombre del archivo de instrucciones:** `Reorganize_instructions_Shore-Test.md`
> **Rama de trabajo:** crear `refactor/docs-and-structure` antes de hacer cualquier cambio; abrir PR hacia `main` al finalizar
> **Idioma de documentación:** español para texto narrativo, inglés para docstrings Python (convención estándar)

---

## 0. Diagnóstico del estado actual

### Inventario completo de archivos y carpetas

| Elemento | Tipo | Estado | Severidad del problema |
|---|---|---|---|
| `py.7z` | Archivo comprimido 7-Zip en raíz | **Versionado en git** ❌ | 🔴 Crítico |
| `README.md` | Documentación | Presente pero con texto de prueba visible | 🔴 Crítico |
| `train_new_classifier.md` | Markdown | Copiado de CoastSat sin atribución | 🟠 Importante |
| `example_jupyter.ipynb` | Notebook de ejemplo | Sin contexto de qué sitio usa | 🟠 Importante |
| `Shore-Test-integrado.ipynb` | Notebook principal | Sin documentación del sitio analizado | 🟠 Importante |
| `funciones_shore.py` | Módulo Python auxiliar | Sin docstrings | 🟠 Importante |
| `data/` | Carpeta de datos satelitales | Sin README; estructura desconocida | 🟡 Deseable |
| `kml/` | Archivos KML de región de interés | Sin README; sin descripción del sitio | 🟡 Deseable |
| `models/` | Clasificadores `.pkl` entrenados | Sin README; sin descripción del entrenamiento | 🟡 Deseable |
| `.gitignore` | Configuración de git | Probablemente ausente | 🔴 Crítico |

### Contexto técnico identificado

**Este repositorio es una adaptación de [CoastSat](https://github.com/kvos/CoastSat)** (Vos et al., 2019), el toolkit de código abierto para extracción de líneas de costa a partir de imágenes satelitales Landsat y Sentinel-2 mediante Google Earth Engine. La arquitectura del repo sigue exactamente la estructura de CoastSat:

- `data/` → imágenes satelitales descargadas de GEE y shapefiles de líneas de costa extraídas
- `kml/` → archivos KML que definen la Región de Interés (ROI) para GEE
- `models/` → clasificadores `.pkl` entrenados (KMeans o SVM) para segmentación arena/agua/espuma/tierra
- `train_new_classifier.md` → documentación de CoastSat copiada directamente, posiblemente adaptada
- `funciones_shore.py` → funciones wrapper o extensiones propias sobre el módulo `coastsat`

El nombre "Shore-Test" y la nota en el README ("Testeando carga de archivos grandes!") indican que este repo nació como un entorno de pruebas, pero el contenido (`Shore-Test-integrado.ipynb`, modelos entrenados, datos reales) sugiere que es un trabajo de investigación aplicada real con resultados concretos.

### Problema más urgente: `py.7z`

El archivo `py.7z` es un archivo comprimido 7-Zip que con alta probabilidad contiene un entorno Python completo, un directorio de datos pesados, o ambos. Su presencia en git junto con la nota "Testeando carga de archivos grandes!" confirma que fue subido como experimento para verificar si git toleraba archivos grandes. **Esto no debe estar versionado en git bajo ninguna circunstancia.**

---

## 1. Tarea previa obligatoria: auditoría del contenido

Ejecutar antes de cualquier modificación. Guardar resultados en `audit_log.txt` (no commitear):

```bash
# 1. Árbol completo con tamaños
find . -not -path './.git/*' -type f -exec ls -lh {} \; | sort -k5 -rh > audit_log.txt

# 2. Tamaño total por carpeta
du -sh */ >> audit_log.txt

# 3. Identificar qué hay dentro de py.7z SIN descomprimirlo
7z l py.7z >> audit_log.txt 2>/dev/null || \
  python3 -c "import subprocess; r = subprocess.run(['7z','l','py.7z'], capture_output=True, text=True); print(r.stdout)" >> audit_log.txt

# 4. Estructura de la carpeta data/
find data/ -type f | head -60 >> audit_log.txt 2>/dev/null

# 5. Estructura de la carpeta kml/
find kml/ -type f >> audit_log.txt 2>/dev/null

# 6. Estructura de la carpeta models/
find models/ -type f >> audit_log.txt 2>/dev/null

# 7. Cabecera y primeras celdas del notebook principal
jupyter nbconvert --to script Shore-Test-integrado.ipynb --stdout 2>/dev/null | head -120 >> audit_log.txt

# 8. Cabecera del notebook de ejemplo
jupyter nbconvert --to script example_jupyter.ipynb --stdout 2>/dev/null | head -80 >> audit_log.txt

# 9. Contenido completo de funciones_shore.py
cat funciones_shore.py >> audit_log.txt

# 10. Contenido del README actual
cat README.md >> audit_log.txt

# 11. Imports de todos los archivos Python
grep -n "^import\|^from" funciones_shore.py >> audit_log.txt 2>/dev/null

# 12. Nombre del sitio costero estudiado
# (buscar en el notebook: variable 'sitename', coordenadas, nombres de playa)
grep -n "sitename\|polygon\|coords\|latitude\|longitude\|playa\|beach\|coast\|chile\|atacama\|valparaiso\|biobio\|antofagasta" \
     funciones_shore.py 2>/dev/null >> audit_log.txt
jupyter nbconvert --to script Shore-Test-integrado.ipynb --stdout 2>/dev/null | \
     grep -in "sitename\|polygon\|coords\|latitude\|longitude\|playa\|beach" >> audit_log.txt
```

Registrar especialmente:
- Qué contiene `py.7z` (entorno Python, datos, modelos, código)
- Nombre del sitio costero (`sitename`) estudiado en el notebook
- Coordenadas geográficas de la ROI (del KML o del notebook)
- Satélites utilizados (Landsat 5/7/8/9, Sentinel-2)
- Rango temporal del análisis
- Versión de CoastSat usada (buscar en imports o en comentarios)

---

## 2. Correcciones críticas e inmediatas

### 2.1 Eliminar `py.7z` del repositorio

Este es el cambio más urgente. El archivo contamina el historial y puede ocupar decenas de MB o GB:

```bash
# Paso 1: ver qué contiene antes de borrarlo
7z l py.7z

# Paso 2: según el contenido, decidir qué hacer con él:
#   - Si es un entorno conda/pip → documentar en requirements.txt y borrar
#   - Si son datos de simulación → mover a data/ si son pequeños, o documentar
#                                   en README dónde conseguirlos si son grandes
#   - Si es código Python → extraer los archivos útiles y versionar solo esos

# Paso 3: quitar del tracking de git
git rm py.7z
# Esto lo elimina del próximo commit pero queda en el historial.
# Si el archivo es grande y se quiere limpiar el historial también:
git filter-repo --path py.7z --invert-paths
# (requiere: pip install git-filter-repo)
# ADVERTENCIA: git filter-repo reescribe el historial; consultar con el dueño
# del repo antes de hacerlo, especialmente si hay colaboradores.
```

> **Decisión para el agente:** si el archivo pesa más de 50 MB y contiene un entorno Python, la acción correcta es `git rm py.7z`, documentar las dependencias en `requirements.txt`, y **no** usar `git filter-repo` sin confirmación explícita del dueño del repo. Dejar una nota en el PR explicando que el historial sigue conteniendo el archivo pero que ya no se trackea.

### 2.2 Crear `.gitignore`

```gitignore
# ─── Python ──────────────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.pyo
.ipynb_checkpoints/
*.egg-info/
dist/ build/
.venv/ env/ venv/

# ─── Archivos comprimidos (nunca versionar entornos o datos grandes así) ─────
*.7z
*.zip
*.tar.gz
*.rar

# ─── Datos satelitales CoastSat (regenerables desde GEE) ─────────────────────
# Imágenes descargadas (pueden ser GB)
data/*/jpg_files/
data/*/ms/
data/*/pan/
data/*/swir/
# Resultados intermedios regenerables
data/*_output.pkl
data/*_metadata.pkl
# Descomentar si se decide no versionar los shapefiles de salida:
# data/*.geojson
# data/*.shp data/*.shx data/*.dbf data/*.prj

# ─── Modelos (versionar solo los .pkl finales entrenados, no intermedios) ─────
# models/*_temp.pkl
# models/*_backup.pkl

# ─── Outputs de figura (regenerables desde los notebooks) ────────────────────
*.jpg *.jpeg *.gif
# Mantener PNGs de referencia si se quieren versionar resultados clave:
# *.png  ← NO descomentar si se quieren versionar figuras de referencia

# ─── Sistema operativo y editores ────────────────────────────────────────────
.DS_Store Thumbs.db *~ *.swp
.vscode/ .idea/
audit_log.txt
```

> **Nota sobre `data/`:** CoastSat descarga imágenes satelitales en subcarpetas por satélite (`jpg_files/`, `ms/`, `pan/`, `swir/`). Estas pueden sumar fácilmente varios GB. Si ya están versionadas, des-trackearlas:
> ```bash
> git rm -r --cached data/*/jpg_files/
> git rm -r --cached data/*/ms/ data/*/pan/ data/*/swir/
> ```
> Los archivos permanecerán localmente pero dejarán de estar en git.

---

## 3. Nueva estructura de directorios propuesta

La estructura actual ya está parcialmente bien organizada siguiendo la convención de CoastSat. La intervención es principalmente aditiva (documentación) con algunos movimientos menores:

```
Shore-Test/
├── README.md                          ← reescribir completamente (§4.1)
├── LICENSE                            ← añadir (MIT recomendado)
├── CITATION.cff                       ← añadir (§5.3)
├── .gitignore                         ← crear (§2.2)
├── requirements.txt                   ← crear (§5.1)
│
├── Shore-Test-integrado.ipynb         ← notebook principal (renombrar, §3.1)
├── example_jupyter.ipynb              ← notebook de referencia de CoastSat
├── funciones_shore.py                 ← módulo auxiliar (documentar, §4.3)
│
├── data/                              ← datos y resultados (añadir README, §4.4)
│   └── README.md
│
├── kml/                               ← regiones de interés (añadir README, §4.5)
│   └── README.md
│
├── models/                            ← clasificadores entrenados (añadir README, §4.6)
│   └── README.md
│
├── doc/                               ← NUEVA: documentación técnica del proyecto
│   ├── site_description.md            ← descripción del sitio costero analizado
│   ├── methods.md                     ← metodología y parámetros de CoastSat usados
│   └── train_new_classifier.md        ← mover aquí desde raíz (§3.2)
│
└── results/                           ← NUEVA: figuras y tablas de referencia
    └── figures/
```

Crear las carpetas nuevas:

```bash
mkdir -p doc results/figures
```

### 3.1 Considerar renombrar el notebook principal

`Shore-Test-integrado.ipynb` es un nombre descriptivo pero contiene "-Test" que sugiere provisionalidad. Evaluar con el dueño si renombrar a algo más definitivo:

```bash
# Opción recomendada (incluye nombre del sitio):
git mv Shore-Test-integrado.ipynb shoreline_analysis_[SITENAME].ipynb

# Alternativa si se quiere preservar el nombre actual:
# No renombrar, solo documentar bien en el README
```

> **Decisión condicional:** si el repo tiene un nombre de sitio claro (p.ej. "Chañaral", "Dichato", "Llico"), incorporarlo en el nombre del notebook. Si es un repo de múltiples sitios, mantener el nombre actual.

### 3.2 Mover `train_new_classifier.md` a `doc/`

```bash
git mv train_new_classifier.md doc/train_new_classifier.md
```

Antes de mover, leer el archivo para verificar si es una copia literal del documento de CoastSat o si tiene modificaciones específicas para este sitio. Actualizar el encabezado según corresponda (ver §4.7).

---

## 4. Documentación

### 4.1 Reescribir `README.md`

**Eliminar inmediatamente la primera línea:** `Testeando carga de archivos grandes!` — esta nota de prueba quedó visible como el primer texto del README del repositorio.

Reemplazar el contenido completo con:

````markdown
# Shore-Test — Análisis de línea de costa con imágenes satelitales

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![CoastSat](https://img.shields.io/badge/Basado_en-CoastSat-green)](https://github.com/kvos/CoastSat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dacastillo/Shore-Test/blob/main/shoreline_analysis_[SITENAME].ipynb)

Aplicación del toolkit [CoastSat](https://github.com/kvos/CoastSat) para el rastreo
de la posición de la línea de costa en **[COMPLETAR: nombre del sitio costero, Chile]**
a partir de imágenes satelitales Landsat y Sentinel-2.

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

- **Ubicación:** [COMPLETAR: nombre de la playa/costa, región, Chile]
- **Coordenadas:** [COMPLETAR: lat/lon del centro de la ROI]
- **Satélites:** [COMPLETAR: Landsat 5 / 7 / 8 / 9 / Sentinel-2]
- **Período analizado:** [COMPLETAR: e.g., 2000–2023]
- **Número de imágenes procesadas:** [COMPLETAR]

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
gcloud auth application-default login
# o alternativamente:
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

## Resultados principales

[COMPLETAR: describir brevemente el resultado más relevante — p.ej.
"La línea de costa en [sitio] retrocedió X metros entre YYYY y YYYY,
con una tasa media de Z m/año."]

![Resultado principal](results/figures/[COMPLETAR].png)

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
````

### 4.2 Crear `doc/site_description.md`

```markdown
# Descripción del sitio costero analizado

## Ubicación geográfica

- **Nombre del sitio:** [COMPLETAR: nombre oficial de la playa/costa]
- **Región:** [COMPLETAR: región administrativa de Chile]
- **Coordenadas del centro de la ROI:** [COMPLETAR: lat, lon en grados decimales]
- **Extensión de la costa analizada:** [COMPLETAR: longitud en km]

## Contexto geomorfológico

[COMPLETAR: descripción del tipo de costa — playa arenosa, costa rocosa,
delta, estuario — y su comportamiento conocido. 3–5 oraciones.]

## Motivación del estudio

[COMPLETAR: por qué este sitio es de interés — erosión costera conocida,
impacto de infraestructura portuaria, efectos de El Niño/La Niña,
cambio climático, petición de institución, etc.]

## Fuente de imágenes satelitales

- **Satélites:** [COMPLETAR]
- **Resolución espacial:** 10 m (Sentinel-2) / 15–30 m (Landsat)
- **Período:** [COMPLETAR: fecha inicio — fecha fin]
- **Número de imágenes disponibles:** [COMPLETAR]
- **Porcentaje de imágenes descartadas por nubosidad:** [COMPLETAR si disponible]

## Archivos KML de la ROI

Ver `kml/README.md` para descripción de cada archivo KML.

## Notas específicas del sitio

[COMPLETAR: particularidades que afectan el análisis —
color inusual de la arena (como Playa Chañaral con arena muy blanca),
presencia de algas, mareas extremas, disponibilidad limitada de imágenes, etc.]
```

### 4.3 Crear `doc/methods.md`

````markdown
# Metodología — Parámetros de CoastSat utilizados

## Versión de CoastSat

[COMPLETAR: versión exacta — `pip show coastsat` o inferir de imports]

## Parámetros de descarga de imágenes (`inputs`)

```python
inputs = {
    'sitename':   '[COMPLETAR]',          # nombre del sitio
    'polygon':    [COMPLETAR],            # coordenadas de la ROI (del KML)
    'dates':      ['[COMPLETAR]', '[COMPLETAR]'],  # rango temporal
    'sat_list':   [COMPLETAR],            # e.g., ['L5','L8','L9','S2']
    'filepath':   'data/',
}
```

## Parámetros de procesamiento (`settings`)

```python
settings = {
    'cloud_thresh':      [COMPLETAR],  # umbral de cobertura de nubes (0–1)
    'dist_clouds':       [COMPLETAR],  # distancia mínima a nubes (m)
    'output_epsg':       [COMPLETAR],  # sistema de referencia de coordenadas
    'check_detection':   [COMPLETAR],  # True/False: verificación manual
    'adjust_detection':  [COMPLETAR],  # True/False: ajuste manual
    'save_figure':       [COMPLETAR],
    'min_beach_area':    [COMPLETAR],  # área mínima de playa detectada (m²)
    'min_length_sl':     [COMPLETAR],  # longitud mínima de línea de costa (m)
    'sand_color':        '[COMPLETAR]',  # 'default','dark','bright','latest'
    'pan_off':           [COMPLETAR],
    's2cloudless_prob':  [COMPLETAR],
}
```

## Clasificador utilizado

- **Archivo:** `models/[COMPLETAR].pkl`
- **Tipo:** [COMPLETAR: KMeans / SVM / otro]
- **Entrenado con:** [COMPLETAR: imágenes de CoastSat por defecto / imágenes
  propias del sitio]
- **Clases:** arena, agua, espuma/rompiente, otras superficies terrestres
- **¿Fue reentrenado para este sitio?** [Sí/No — si sí, ver `doc/train_new_classifier.md`]

## Transectos para análisis temporal

[COMPLETAR: descripción de los transectos definidos — número, orientación,
longitud, distancia entre ellos]

## Corrección de marea (si aplica)

[COMPLETAR: ¿se aplicó corrección de marea? Si sí, modelo usado (FES2022 u otro),
rango de marea en el sitio]
````

### 4.4 `data/README.md`

```markdown
# Carpeta de datos — `data/`

Contiene las imágenes satelitales descargadas de Google Earth Engine y
los resultados del análisis de línea de costa generados por CoastSat.

## ⚠️ Archivos NO versionados en git

Las imágenes satelitales crudas (subcarpetas `jpg_files/`, `ms/`, `pan/`, `swir/`)
**no se incluyen en este repositorio** por su tamaño (pueden superar varios GB).
Para regenerarlas, ejecutar las celdas de descarga en el notebook principal.

## Archivos SÍ versionados

| Archivo | Descripción |
|---------|-------------|
| `[sitename]_output.pkl` | Líneas de costa extraídas (fechas + coordenadas) |
| `[sitename]_output.geojson` | Líneas de costa en formato GeoJSON (para QGIS/GEE) |
| `[sitename]_reference_shoreline.pkl` | Línea de costa de referencia para validación |
| `[sitename]_transect_time_series.csv` | Serie temporal de posición por transecto |
| [COMPLETAR otros archivos encontrados] | |

## Cómo regenerar los datos

```bash
# Activar entorno y abrir notebook
conda activate coastsat
jupyter notebook Shore-Test-integrado.ipynb

# Ejecutar Sección 1: Descarga de imágenes
# (requiere autenticación con Google Earth Engine activa)
```

## Sistema de referencia de coordenadas

EPSG: [COMPLETAR — inferir de `settings['output_epsg']` en el notebook]
```

### 4.5 `kml/README.md`

```markdown
# Archivos KML — Región de Interés (ROI)

Contiene los archivos KML que definen el polígono de la región de interés
utilizada por Google Earth Engine para descargar las imágenes satelitales.

## Archivos

| Archivo | Descripción | Coordenadas aproximadas |
|---------|-------------|------------------------|
| [COMPLETAR: nombre del .kml] | [COMPLETAR: sitio que define] | [lat, lon] |

## Cómo usar estos archivos

Los archivos KML se cargan en el notebook principal para definir el polígono
de la ROI. También pueden abrirse en Google Earth o QGIS para visualización.

```python
# Ejemplo de cómo se usa en el notebook (CoastSat):
from coastsat import SDS_tools
polygon = SDS_tools.polygon_from_kml('kml/[nombre].kml')
```

## Cómo crear o modificar la ROI

Para definir una nueva ROI, dibujar el polígono en
[geojson.io](https://geojson.io) o en Google Earth y exportar como KML.
El polígono debe encerrar completamente la franja de costa a analizar,
incluyendo al menos 500 m tierra adentro y 500 m mar adentro.
```

### 4.6 `models/README.md`

```markdown
# Clasificadores entrenados — `models/`

Contiene los archivos `.pkl` de los clasificadores de píxeles usados por
CoastSat para segmentar las imágenes en cuatro categorías:
arena · agua · espuma/rompiente · otras superficies terrestres.

## Archivos

| Archivo `.pkl` | Origen | Optimizado para | Fecha de entrenamiento |
|----------------|--------|-----------------|------------------------|
| [COMPLETAR] | CoastSat por defecto / entrenamiento propio | [tipo de playa] | [COMPLETAR] |

## Cuándo usar cada clasificador

- **Clasificador por defecto de CoastSat:** para playas arenosas de color
  intermedio. Activar con `settings['sand_color'] = 'default'`.
- **Clasificador `dark`:** para playas de arena gris/negra (arena volcánica).
- **Clasificador `bright`:** para playas de arena muy blanca (Atacama, coral).
- **Clasificador propio (si existe):** entrenado con imágenes de este sitio
  específico. Ver `doc/train_new_classifier.md`.

## Cómo reentrenar

Ver `doc/train_new_classifier.md` para instrucciones completas.

El clasificador reentrenado se guarda automáticamente en esta carpeta como
un nuevo archivo `.pkl`.
```

### 4.7 Revisar y atribuir `doc/train_new_classifier.md`

Leer el contenido del archivo. Según lo encontrado:

**Si es una copia literal de `CoastSat/doc/train_new_classifier.md`:**
Añadir al inicio del archivo:

```markdown
> **Nota:** Este documento es una copia del tutorial oficial de CoastSat
> ([kvos/CoastSat](https://github.com/kvos/CoastSat/blob/master/doc/train_new_classifier.md))
> con adaptaciones para el sitio [COMPLETAR nombre del sitio].
> Cambios propios respecto al original: [COMPLETAR o "ninguno"].
```

**Si tiene modificaciones propias:** documentar explícitamente qué se cambió
respecto al tutorial original de CoastSat.

### 4.8 Documentar `funciones_shore.py`

Leer el archivo completo. Para cada función, añadir docstring en formato NumPy:

```python
"""
funciones_shore.py — Funciones auxiliares para el análisis de línea de costa.

Extensiones y wrappers sobre el módulo coastsat para el análisis específico
del sitio [COMPLETAR: nombre del sitio].

Funciones principales:
    - [COMPLETAR: listar funciones encontradas]

Uso:
    from funciones_shore import [función_principal]

Dependencias:
    - coastsat
    - numpy, scipy, matplotlib
    - geopandas (si maneja geometrías)

Autor: Daniel Castillo-Castro
"""
```

Para cada función individual, usar la plantilla:

```python
def nombre_funcion(param1, param2):
    """Descripción en una línea.

    Descripción extendida del propósito, indicando si es un wrapper
    de una función de CoastSat o una extensión propia.

    Parameters
    ----------
    param1 : tipo
        Descripción. Incluir unidades si aplica (e.g., metros, días).
    param2 : tipo
        Descripción.

    Returns
    -------
    tipo
        Descripción del valor de retorno.

    Notes
    -----
    Si wrappea una función de CoastSat, indicar cuál:
    e.g., "Wrapper de coastsat.SDS_shoreline.extract_shorelines()"
    """
```

---

## 5. Archivos de configuración y soporte

### 5.1 Crear `requirements.txt`

Construir a partir de los imports encontrados en la auditoría (§1):

```bash
# Si el entorno está activo:
pip freeze | grep -E "coastsat|earthengine|geopandas|scikit-image|matplotlib|\
numpy|scipy|astropy|pyqt5|imageio|shapely|fiona|pyproj|pandas" > requirements.txt

# Si no, construir manualmente:
cat > requirements.txt << 'EOF'
# CoastSat y dependencias geoespaciales
coastsat>=2.5.0
earthengine-api>=0.1.360
geopandas>=0.13.0
scikit-image>=0.21.0
shapely>=2.0.0
fiona>=1.9.0
pyproj>=3.6.0

# Científico general
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0
astropy>=5.3.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0

# Visualización interactiva
pyqt5>=5.15.0

# Para corrección de mareas (CoastSat >= 3.0)
# pyfes>=2.9.3   # instalar via conda-forge, no pip

# [COMPLETAR con otros imports encontrados en funciones_shore.py]
EOF
```

> **Nota sobre `pyfes`:** la librería de corrección de mareas FES2022 debe instalarse via `conda install conda-forge::pyfes`, no con pip. Documentarlo en el README.

### 5.2 Añadir `LICENSE`

```bash
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2023 Daniel Castillo-Castro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
EOF
```

### 5.3 Añadir `CITATION.cff`

```yaml
cff-version: 1.2.0
message: "Si usas este código, cita tanto este repo como CoastSat."
type: software
authors:
  - family-names: Castillo-Castro
    given-names: Daniel
    affiliation: "Universidad Mayor, Chile"
title: "Shore-Test: Análisis de línea de costa con CoastSat en [COMPLETAR sitio], Chile"
version: "1.0.0"
date-released: "2023-11-01"   # COMPLETAR con fecha real
url: "https://github.com/Dacastillo/Shore-Test"
license: MIT
keywords:
  - shoreline
  - coastal monitoring
  - satellite remote sensing
  - Google Earth Engine
  - CoastSat
  - Chile
references:
  - type: software
    authors:
      - family-names: Vos
        given-names: Kilian
    title: "CoastSat"
    url: "https://github.com/kvos/CoastSat"
  - type: article
    authors:
      - family-names: Vos
        given-names: Kilian
    title: "CoastSat: A Google Earth Engine-enabled Python toolkit to extract
            shorelines from publicly available satellite imagery"
    journal: "Environmental Modelling & Software"
    year: 2019
    doi: "10.1016/j.envsoft.2019.104528"
```

---

## 6. Mejoras opcionales

### 6.1 Añadir visualización de resultados en el README

Incluir en el README una figura de ejemplo del resultado del análisis.
Si el notebook ya genera figuras, exportar la más representativa:

```bash
# Exportar figura desde el notebook ejecutado
jupyter nbconvert --to html --execute Shore-Test-integrado.ipynb \
  --output results/Shore-Test_results.html
```

O extraer una imagen específica y guardarla en `results/figures/`.

### 6.2 Añadir botón de Binder

Si el notebook puede ejecutarse sin acceso a GEE (usando los datos ya descargados
en `data/`), añadir un badge de Binder al README. Si requiere GEE
(lo más probable), usar Google Colab en su lugar — ya incluido en la plantilla
de README de §4.1.

### 6.3 Topics recomendados para GitHub

Añadir en Settings → Topics:
`shoreline-detection`, `coastal-monitoring`, `remote-sensing`,
`google-earth-engine`, `coastsat`, `satellite-imagery`, `landsat`,
`sentinel-2`, `python`, `jupyter-notebook`, `chile`, `geospatial`

### 6.4 Considerar renombrar el repositorio

El nombre `Shore-Test` transmite provisionalidad. Si el análisis es un trabajo
de investigación real con resultados publicables, considerar renombrar a algo
como `shoreline-[SITENAME]-chile` o `coastsat-[SITENAME]`.
El renombrado se hace en GitHub → Settings → Repository name.
Git actualiza automáticamente los remotes de los clones existentes con un redirect.

### 6.5 Vincular con `Dacastillo/r`

Si el análisis o post-procesado tiene scripts adicionales en el repo `r`,
añadir una mención cruzada en el README:

```markdown
## Análisis extendido

Scripts de post-procesado adicionales disponibles en
[Dacastillo/r — python/analysis/](https://github.com/Dacastillo/r/tree/main/python/analysis/).
```

---

## 7. Orden de ejecución recomendado

| Paso | Tarea | Sección | Prioridad |
|------|-------|---------|-----------|
| 1 | Crear rama `refactor/docs-and-structure` | — | — |
| 2 | Ejecutar auditoría completa y leer `audit_log.txt` | §1 | **Obligatorio antes de todo** |
| 3 | Inspeccionar contenido de `py.7z` | §2.1 | 🔴 Crítico |
| 4 | Eliminar `py.7z` con `git rm` | §2.1 | 🔴 Crítico |
| 5 | Crear `.gitignore` y des-trackear datos satelitales si están versionados | §2.2 | 🔴 Crítico |
| 6 | Eliminar la línea "Testeando carga de archivos grandes!" del README | §4.1 | 🔴 Crítico |
| 7 | Crear carpetas `doc/` y `results/figures/` | §3 | 🟠 Importante |
| 8 | Mover `train_new_classifier.md` a `doc/` | §3.2 | 🟠 Importante |
| 9 | Reescribir `README.md` completo | §4.1 | 🟠 Importante |
| 10 | Crear `doc/site_description.md` | §4.2 | 🟠 Importante |
| 11 | Crear `doc/methods.md` con parámetros reales del notebook | §4.3 | 🟠 Importante |
| 12 | Añadir READMEs a `data/`, `kml/`, `models/` | §4.4–4.6 | 🟠 Importante |
| 13 | Revisar y atribuir `doc/train_new_classifier.md` | §4.7 | 🟠 Importante |
| 14 | Documentar `funciones_shore.py` con docstrings | §4.8 | 🟡 Deseable |
| 15 | Crear `requirements.txt` | §5.1 | 🟠 Importante |
| 16 | Añadir `LICENSE` | §5.2 | 🟡 Deseable |
| 17 | Añadir `CITATION.cff` | §5.3 | 🟡 Deseable |
| 18 | Considerar renombrar notebook y/o repo | §3.1, §6.4 | 🟢 Opcional |
| 19 | Commit: `refactor: remove large file, add full documentation and gitignore` | — | — |
| 20 | Abrir PR hacia `main` con descripción de todos los cambios | — | — |

---

## 8. Restricciones y advertencias

- **`py.7z` en el historial:** `git rm py.7z` lo quita del próximo commit pero permanece en el historial de git. Si el archivo es muy grande (>100 MB), el historial seguirá siendo pesado. Limpiar el historial con `git filter-repo` es posible pero **reescribe los hashes de todos los commits** — consultar con el dueño antes de hacerlo, ya que invalida cualquier fork o clone existente.
- **No modificar los datos en `data/`:** los archivos `.pkl`, `.geojson` y `.csv` de resultados son salidas científicas — solo documentarlos, nunca editarlos.
- **No modificar los modelos en `models/`:** los archivos `.pkl` son pesos entrenados — solo documentarlos.
- **Los archivos KML en `kml/` definen la ciencia:** cambiarlos significaría cambiar el área de estudio. Solo documentarlos.
- **Atribución de CoastSat:** cualquier publicación o presentación que use este código debe citar a Vos et al. (2019). Asegurarse de que el `CITATION.cff` lo refleje.
- **No hacer `--force-push` sobre `main`** bajo ninguna circunstancia.
- **No reescribir historial** (`git filter-repo` o `git rebase -i`) sin confirmación explícita del dueño del repo.

---

## 9. Criterio de éxito

El trabajo está completo cuando:

- [ ] `py.7z` no aparece en `git status` ni en `git ls-files`
- [ ] El `README.md` no comienza con "Testeando carga de archivos grandes!"
- [ ] `README.md` describe el sitio costero, el método, el período y cómo reproducir el análisis, sin campos `[COMPLETAR]` sin resolver
- [ ] `.gitignore` excluye imágenes satelitales, archivos comprimidos y cachés de Python/Jupyter
- [ ] `doc/site_description.md` identifica claramente el sitio geográfico estudiado
- [ ] `doc/methods.md` documenta todos los parámetros de CoastSat utilizados
- [ ] `data/README.md`, `kml/README.md` y `models/README.md` describen el contenido de cada carpeta
- [ ] `funciones_shore.py` tiene docstring de módulo y docstring para cada función
- [ ] `requirements.txt` lista las dependencias con versiones
- [ ] `doc/train_new_classifier.md` atribuye correctamente su origen en CoastSat
- [ ] El PR está abierto con descripción clara de todos los cambios realizados

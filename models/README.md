# Clasificadores entrenados — `models/`

Contiene los archivos `.pkl` de los clasificadores de píxeles usados por
CoastSat para segmentar las imágenes en cuatro categorías:
**arena**, **agua**, **espuma/rompiente** y **otras superficies terrestres**.

## Clasificadores de Red Neuronal (MLP)

### Para Landsat

| Archivo | Descripción | Uso recomendado |
|---------|-------------|-----------------|
| `NN_4classes_Landsat.pkl` | Clasificador base | Playas de arena estándar |
| `NN_4classes_Landsat_new.pkl` | Versión actualizada | Uso general |
| `NN_4classes_Landsat_dark.pkl` | Arena oscura | Playas volcánicas, arena gris |
| `NN_4classes_Landsat_bright.pkl` | Arena clara | Playas de arena blanca, coral |
| `NN_4classes_Landsat_test.pkl` | Versión de prueba | Testing y validación |
| `NN_4classes_Landsat_latest_new.pkl` | Versión más reciente | Uso recomendado |

### Para Sentinel-2

| Archivo | Descripción | Uso recomendado |
|---------|-------------|-----------------|
| `NN_4classes_S2.pkl` | Clasificador base S2 | Uso general |
| `NN_4classes_S2_new.pkl` | Versión actualizada S2 | Uso recomendado |

## Conjuntos de entrenamiento

| Archivo | Satélite | Descripción |
|---------|----------|-------------|
| `CoastSat_training_set_L8.pkl` | Landsat 8 | Dataset de entrenamiento original |
| `CoastSat_training_set_S2.pkl` | Sentinel-2 | Dataset de entrenamiento S2 |

## Cuándo usar cada clasificador

- **Clasificador por defecto (`default`):** Para playas arenosas de color intermedio.
  Activar con `settings['sand_color'] = 'default'`.

- **Clasificador `dark`:** Para playas de arena gris/negra (arena volcánica,
  minerales oscuros). Activar con `settings['sand_color'] = 'dark'`.

- **Clasificador `bright`:** Para playas de arena muy blanca (Atacama, coral,
  cuarzo puro). Activar con `settings['sand_color'] = 'bright'`.

- **Clasificador `latest`:** La versión más actualizada, recomendada para
  uso general.

## Arquitectura del clasificador

Los clasificadores son redes neuronales Multilayer Perceptron (MLP) con:

- **Capas ocultas:** 2 capas (100 y 50 neuronas)
- **Función de activación:** ReLU
- **Optimizador:** Adam
- **Clases de salida:** 4 (arena, agua, espuma, otros)

## Cómo reentrenar

Ver `doc/train_new_classifier.md` para instrucciones completas sobre cómo:

1. Etiquetar nuevas imágenes del sitio de estudio
2. Entrenar un nuevo clasificador
3. Evaluar la precisión del clasificador
4. Guardar el nuevo modelo `.pkl`

## Precisión típica

Los clasificadores pre-entrenados de CoastSat típicamente alcanzan:

- **Precisión global:** 85-95%
- **Precisión para arena:** 90-98%
- **Precisión para agua:** 95-99%

La precisión puede variar según las características específicas del sitio.

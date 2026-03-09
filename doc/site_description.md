# Descripción del sitio costero analizado

## Ubicación geográfica

Este repositorio contiene análisis para múltiples sitios costeros en Nueva Gales del Sur, Australia:

- **Sitios principales:**
  - **Sawtell:** Costa norte de NSW, área residencial con playas arenosas
  - **Newcastle:** Ciudad costera con puertos y playas urbanas
  - **Byron Bay:** Punto más oriental de Australia, conocido por sus playas y turismo

- **Coordenadas:** Ver archivos KML individuales en `kml/` para los polígonos exactos de cada sitio
  - Sawtell: aproximadamente -30.36°S, 153.10°E
  - Newcastle: aproximadamente -32.93°S, 151.78°E
  - Byron Bay: aproximadamente -28.64°S, 153.60°E

## Contexto geomorfológico

La costa de Nueva Gales del Sur se caracteriza por:

- Playas de arena de longitud variable
- Influencia de mareas moderadas
- Exposición al oleaje del Océano Pacífico
- Variabilidad estacional en la posición de la línea de costa

## Motivación del estudio

El análisis de la dinámica costera en estos sitios es relevante para:

- Monitoreo de erosión costera
- Gestión de zonas costeras urbanizadas
- Estudio de impactos de tormentas y eventos extremos
- Planificación costera a largo plazo

## Fuente de imágenes satelitales

- **Satélites:** Landsat 5, 7, 8, 9 y Sentinel-2
- **Resolución espacial:** 10-30 m según el sensor
- **Período:** Ver notebook principal para el rango específico analizado

## Archivos KML de la ROI

Cada sitio tiene su propio archivo KML en la carpeta `kml/`:

| Archivo | Sitio | Coordenadas aproximadas |
|---------|-------|------------------------|
| SAWTELL.kml | Sawtell, NSW | -30.36°S, 153.10°E |
| NEWCASTLE.kml | Newcastle, NSW | -32.93°S, 151.78°E |
| BYRON.kml | Byron Bay, NSW | -28.64°S, 153.60°E |

## Notas específicas del sitio

- Los archivos KML definen polígonos que incluyen la franja costera y una zona de amortiguamiento mar adentro y tierra adentro
- Las imágenes pueden verse afectadas por cobertura nubosa, especialmente durante meses de invierno
- La presencia de estructuras portuarias en Newcastle puede requerir ajustes en el procesamiento

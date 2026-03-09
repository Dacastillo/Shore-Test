# Archivos KML — Región de Interés (ROI)

Contiene los archivos KML que definen el polígono de la región de interés
utilizada por Google Earth Engine para descargar las imágenes satelitales.

## Archivos disponibles

| Archivo | Sitio | Coordenadas aproximadas |
|---------|-------|------------------------|
| `SAWTELL.kml` | Sawtell, NSW, Australia | -30.36°S, 153.10°E |
| `NEWCASTLE.kml` | Newcastle, NSW, Australia | -32.93°S, 151.78°E |
| `BYRON.kml` | Byron Bay, NSW, Australia | -28.64°S, 153.60°E |

## Cómo usar estos archivos

Los archivos KML se cargan en el notebook principal para definir el polígono
de la ROI. También pueden abrirse en Google Earth o QGIS para visualización.

### Ejemplo de uso en Python

```python
from coastsat import SDS_tools

# Cargar polígono desde KML
polygon = SDS_tools.polygon_from_kml('kml/SAWTELL.kml')

# Usar en la configuración de descarga
inputs = {
    'sitename': 'SAWTELL',
    'polygon': polygon,
    'dates': ['2020-01-01', '2023-12-31'],
    'sat_list': ['L8', 'S2'],
    'filepath': 'data/',
}
```

## Estructura del KML

Cada archivo KML contiene un polígono cerrado definido por coordenadas
(longitud, latitud, altitud). El polígono debe:

1. Encerrar completamente la franja de costa a analizar
2. Incluir al menos 500 m tierra adentro
3. Incluir al menos 500 m mar adentro
4. Ser aproximadamente rectangular, alineado con la costa

## Cómo crear o modificar la ROI

Para definir una nueva ROI:

1. **Usando geojson.io:**
   - Abrir [geojson.io](https://geojson.io)
   - Dibujar el polígono sobre el mapa
   - Exportar como KML

2. **Usando Google Earth:**
   - Abrir Google Earth
   - Usar la herramienta de polígono
   - Guardar como KML

3. **Usando QGIS:**
   - Crear nueva capa de polígono
   - Digitalizar el área de interés
   - Exportar como KML

## Notas

- Las coordenadas están en WGS84 (EPSG:4326)
- El orden de coordenadas es: longitud, latitud, altitud
- El polígono debe estar cerrado (primer y último punto coinciden)

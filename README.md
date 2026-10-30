# Incidentes viales en Medellín (2008–2025): exploración, predicción y agrupamiento
Proyecto de machine learning con scikit-learn sobre 702.540 incidentes de tránsito registrados en Medellín. Incluye una exploración con mapas de calor, modelos de aprendizaje supervisado (predecir la gravedad de un incidente) y modelos no supervisados (zonas, grupos de comunas y focos de incidentes fatales).

Todos los resultados de este documento salen de ejecutar los scripts de este repositorio.

## Preguntas que busca responder
1. ¿Cuándo y dónde ocurren los incidentes?
2. ¿Se puede predecir si un incidente deja víctimas? ¿Y si es fatal?
3. ¿Qué zonas y comunas se parecen entre sí? ¿Dónde se agrupan los incidentes fatales?

## Datos
  * Archivo: Fatal_Road_Traffic.xlsx (702.540 registros, años 2008–2025).
  * Fuente: [Dataset on Road Traffic Fatalities in Medellín, Colombia (2008–2025)](https://data.mendeley.com/datasets/r6g5dfnpgh/1).
  * Columnas usadas: fecha, hora, clase de incidente, gravedad, latitud, longitud, comuna y barrio.
  * Gravedad: 57,0 % heridos, 42,6 % solo daños, 0,39 % muertos.
  * Datos vacíos: latitud y longitud 6,4 %, comuna 4,0 %, barrio 6,7 %.

> [!IMPORTANT]
> El archivo de datos y el caché no está incluido en el repositorio por su tamaño. Colócalo en datos/Fatal_Road_Traffic.xlsx.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```text
traffic_accidents_medallo/
├── datos/                  # Excel original y caché .parquet (no se suben a GitHub)
├── figuras/                # Gráficas generadas por los scripts
├── src/                    
│   ├── limpieza.py         # Carga, limpieza y guardado de figuras (lo usan los demás)
│   ├── 01analisis.py       # Análisis exploratorio con mapas de calor
│   ├── 02analisissupervisado.py # Modelos de clasificación
│   └── 03analisisnosupervisado.py # K-Means, PCA y DBSCAN
├── .gitignore
├── requirements.txt
└── README.md
```
## Cómo ejecutarlo
Requiere Python 3.12. Desde la carpeta raíz del proyecto (ejemplo en Windows PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/01analisis.py
python src/03analisisnosupervisado.py
python src/02analisissupervisado.py
```
## Resultados
1. **¿Cuándo y dónde ocurren los incidentes?**
De lunes a viernes, alrededor de las 7 a.m. y las 5 p.m.; el sábado, hacia las 13 h; el domingo es el día más claro.[fig 01 hora dia](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/figuras/01_heatmap_hora_dia.png)
Los incidentes forman una franja alargada de norte a sur, más intensa en el centro, y los fatales se alinean en una franja vertical en esa zona central.[fig 04 mapa calor geografico](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/figuras/04_mapa_calor_geografico.png)
La Candelaria es la comuna con más incidentes por año entre las 12 graficadas. [fig 03 comuna year](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/figuras/03_heatmap_comuna_year.png)
El atropello es la clase con mayor porcentaje de muertes (1,3 %) y el choque la que tiene más "solo daños" (64,4 %).[fig 02 incidente gravedad](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/figuras/02_heatmap_incidente_gravedad.png)

2. **¿Se puede predecir la gravedad?**

  * Se plantearon dos tareas: A (¿hubo víctimas o solo daños?, 57 % de casos positivos) y B (¿fue fatal?, 0,39 % de casos positivos). Se compararon dos modelos Dummy de referencia (que no usan las variables de entrada: uno responde siempre la clase más frecuente y el otro responde al azar según las proporciones de cada clase), regresión logística, árbol de decisión y Random Forest, entrenando con el 75 % de los datos y probando con el 25 % restante.

3. **¿Qué se parece entre sí y dónde se agrupan los fatales?**
  * Zonas (K-Means, k = 8): las zonas con más choques tienen más incidentes de "solo daños", y la zona con más atropellos tiene más heridos.
  * Comunas (PCA + K-Means): se forman cuatro grupos: siete comunas de mayor volumen y más choques (La Candelaria, El Poblado, Laureles Estadio, Belén, Castilla, Guayabal y La América), doce con más atropellos e incidentes nocturnos, y los corregimientos de San Sebastián de Palmitas y Santa Elena, que quedan cada uno en su propio grupo. Los dos primeros componentes del PCA explican el 58 % y el 21 % de la varianza.[fig 12 pca comunas](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/figuras/12_pca_comunas.png)
  * Fatales (DBSCAN, 300 m, mínimo 5 puntos): 42 focos y 12 % de puntos como ruido. Un foco grande recorre el centro del mapa de norte a sur; los demás son pequeños y están hacia los costados.[fig 14 dbscan](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/figuras/14_dbscan_fatales.png)

    ## Limitaciones
* Tarea B: con solo 0,39 % de incidentes fatales y estas variables, los modelos obtuvieron resultados muy bajos.
* Partición aleatoria: los modelos se entrenaron y probaron con datos mezclados de todos los años; no se evaluó su comportamiento con años posteriores.
* Elección de k: el silhouette fue más alto para k = 4 (0,404) y k = 5 (0,402) que para k = 8 (0,369); se eligió 8 por nivel de detalle.
* DBSCAN: con 300 m, un foco grande agrupa gran parte de los puntos del centro, así que no separa lugares concretos dentro de esa zona.

## Tecnologías
* Python 3.12 · pandas · NumPy · scikit-learn · matplotlib · seaborn · openpyxl · pyarrow (versiones exactas en [requirements.txt](https://github.com/GandalfTercero/accidentes_transito_med/blob/main/requirements.txt)).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from limpieza import load_data, savefig, DIAS

sns.set_theme(style="white")
df = load_data()

#Vistazo general
print("Filas y columnas:", df.shape)
print()
print("Proporción de cada gravedad:")
print(df["gravedad"].value_counts(normalize=True).round(4))
print()
print("Proporción de datos vacíos por columna:")
print(df.isna().mean().round(3))

# Mapa de calor: hora x día de la semana (¿cuándo ocurren los incidentes?)
# crosstab cuenta cuántas filas hay en cada combinación (día, hora)
tabla = pd.crosstab(df["weekday"], df["hour"])
tabla.index = DIAS   

plt.figure(figsize=(12, 4))
sns.heatmap(tabla, cmap="YlOrRd", cbar_kws={"label": "# incidentes"})
plt.title("Incidentes por hora y día de la semana")
plt.xlabel("Hora")
plt.ylabel("")
savefig("01_heatmap_hora_dia")

# Mapa de calor: clase de incidente x gravedad
tabla = pd.crosstab(df["clase"], df["gravedad"], normalize="index") * 100

plt.figure(figsize=(6, 4))
sns.heatmap(tabla, annot=True, fmt=".1f", cmap="Blues")
plt.title("Gravedad según la clase de incidente (%)")
plt.ylabel("")
savefig("02_heatmap_incidente_gravedad")

# Mapa de calor: comuna x año (solo las 12 comunas con más accidentes)
top_comunas = df["comuna"].value_counts().head(12).index
df_top = df[df["comuna"].isin(top_comunas)]
tabla = pd.crosstab(df_top["comuna"], df_top["year"])

plt.figure(figsize=(12, 5))
sns.heatmap(tabla, cmap="mako_r", cbar_kws={"label": "# incidentes"})
plt.title("Incidentes por comuna y año (top 12 comunas)")
plt.ylabel("")
savefig("03_heatmap_comuna_year")

# Mapa de calor geográfico: dónde se concentran los incidentes
con_coords = df.dropna(subset=["lat", "lon"])
solo_fatales = con_coords[con_coords["gravedad"] == "MUERTO"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# hexbin divide el mapa en hexágonos y colorea según cuántos puntos caen en cada uno
mapa1 = ax1.hexbin(con_coords["lon"], con_coords["lat"], gridsize=60,
                   cmap="inferno", mincnt=1, bins="log")
ax1.set_title("Todos los incidentes")
ax1.set_xlabel("Longitud")
ax1.set_ylabel("Latitud")
plt.colorbar(mapa1, ax=ax1, label="log10(conteo)")

mapa2 = ax2.hexbin(solo_fatales["lon"], solo_fatales["lat"], gridsize=60,
                   cmap="inferno", mincnt=1, bins="log")
ax2.set_title("Solo fatales")
ax2.set_xlabel("Longitud")
ax2.set_ylabel("Latitud")
plt.colorbar(mapa2, ax=ax2, label="log10(conteo)")
savefig("04_mapa_calor_geografico")

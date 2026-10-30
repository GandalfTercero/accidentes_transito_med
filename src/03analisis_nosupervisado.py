# Aprendizaje NO supervisado: Busco grupos en los datos.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from limpieza import load_data, savefig, SEED

sns.set_theme(style="white")
df = load_data()

# K-Means sobre las coordenadas: dividir la ciudad en zonas
con_coords = df.dropna(subset=["lat", "lon"]).copy()
coordenadas = con_coords[["lat", "lon"]]  

# ¿Cuántas zonas (k) conviene usar? Probamos varios k con una muestra y miro el
# "silhouette" (entre más alto, mejor separados quedan los grupos)
muestra = coordenadas.sample(20000, random_state=SEED)
print("Silhouette para cada k:")
for k in range(3, 11):
    modelo = KMeans(n_clusters=k, n_init=5, random_state=SEED)
    etiquetas = modelo.fit_predict(muestra)
    print("  k =", k, "->", round(silhouette_score(muestra, etiquetas), 3))

K = 8
kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED)
con_coords["zona"] = kmeans.fit_predict(coordenadas)

plt.figure(figsize=(7, 7))
plt.scatter(con_coords["lon"], con_coords["lat"], c=con_coords["zona"], cmap="tab10", s=1, alpha=0.3)
plt.title("K-Means (k=" + str(K) + ") sobre coordenadas")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
savefig("10_kmeans_zonas")

# ¿Qué tiene de particular cada zona? Mapa de calor con su perfil
con_coords["noche"] = con_coords["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])

por_clase = pd.crosstab(con_coords["zona"], con_coords["clase"], normalize="index")
por_gravedad = pd.crosstab(con_coords["zona"], con_coords["gravedad"], normalize="index")
perfil = pd.concat([por_clase, por_gravedad], axis=1)
perfil["% noche"] = con_coords.groupby("zona")["noche"].mean()

# Para el color usamos el z-score (qué tan por encima o debajo del promedio está cada zona)
# y en cada casilla escribimos la proporción real
perfil_z = (perfil - perfil.mean()) / perfil.std()

plt.figure(figsize=(11, 4.5))
sns.heatmap(perfil_z, annot=perfil.round(2), fmt="", cmap="coolwarm", center=0,
            cbar_kws={"label": "z-score entre zonas"})
plt.title("Perfil de las zonas (color = z-score, número = proporción)")
savefig("11_heatmap_perfil_zonas")

# Agrupar las COMUNAS según su "personalidad": PCA + K-Means
con_comuna = df.dropna(subset=["comuna"]).copy()
con_comuna["noche"] = con_comuna["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
con_comuna["finde"] = con_comuna["weekday"] >= 5
con_comuna["es_fatal"] = (con_comuna["gravedad"] == "MUERTO")

# Una fila por comuna, una columna por característica
tabla = pd.crosstab(con_comuna["comuna"], con_comuna["clase"], normalize="index")
tabla["pct_fatal"] = con_comuna.groupby("comuna")["es_fatal"].mean()
tabla["pct_noche"] = con_comuna.groupby("comuna")["noche"].mean()
tabla["pct_finde"] = con_comuna.groupby("comuna")["finde"].mean()
tabla["log_volumen"] = np.log1p(con_comuna["comuna"].value_counts())   # log para que no domine el volumen
tabla = tabla.dropna()
tabla = tabla.loc[:, tabla.std() > 0]   # quitamos columnas que valen lo mismo en todas las comunas

# Escalamos para que todas las columnas pesen igual
escalador = StandardScaler()
Z = escalador.fit_transform(tabla)

# PCA resume todas las columnas en solo 2 para poder dibujarlas
pca = PCA(n_components=2, random_state=SEED)
puntos = pca.fit_transform(Z)
print("Varianza explicada por PC1 y PC2:", pca.explained_variance_ratio_.round(3))

# K-Means agrupa las comunas en 4 grupos
kmeans_comunas = KMeans(n_clusters=4, n_init=20, random_state=SEED)
grupos = kmeans_comunas.fit_predict(Z)

plt.figure(figsize=(8, 6))
plt.scatter(puntos[:, 0], puntos[:, 1], c=grupos, cmap="tab10", s=80)
nombres_comunas = list(tabla.index)
for i in range(len(nombres_comunas)):
    plt.annotate(nombres_comunas[i], (puntos[i, 0], puntos[i, 1]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.xlabel("PC1 (" + str(round(pca.explained_variance_ratio_[0] * 100)) + "%)")
plt.ylabel("PC2 (" + str(round(pca.explained_variance_ratio_[1] * 100)) + "%)")
plt.title("Comunas en 2D (PCA), coloreadas por grupo K-Means")
savefig("12_pca_comunas")

# Mapas de calor: ¿qué significa cada PC? ¿y cada grupo de comunas?
cargas = pd.DataFrame(pca.components_.T, index=tabla.columns, columns=["PC1", "PC2"])

tabla["grupo"] = grupos
promedios = tabla.groupby("grupo").mean()
promedios_z = (promedios - promedios.mean()) / promedios.std()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cargas, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False, ax=ax1)
ax1.set_title("Cargas del PCA")
sns.heatmap(promedios_z.T, annot=True, fmt=".1f", cmap="coolwarm", center=0,
            cbar_kws={"label": "z-score"}, ax=ax2)
ax2.set_title("Perfil de cada grupo de comunas")
savefig("13_heatmap_pca_clusters")

# DBSCAN: encontrar focos de accidentes fatales
fatales = df[df["gravedad"] == "MUERTO"].dropna(subset=["lat", "lon"]).copy()

# Con metric="haversine" las coordenadas deben estar en radianes
coords_rad = np.radians(fatales[["lat", "lon"]].values)

# Un foco = al menos 5 accidentes fatales a menos de 300 metros entre sí.
# eps se da en radianes: distancia en km / radio de la Tierra (6371 km)
dbscan = DBSCAN(eps=0.3 / 6371, min_samples=5, metric="haversine")
fatales["foco"] = dbscan.fit_predict(coords_rad)   # -1 significa "ruido" (no pertenece a ningún foco)

n_focos = fatales["foco"].max() + 1
pct_ruido = (fatales["foco"] == -1).mean()
print("DBSCAN:", n_focos, "focos y", round(pct_ruido * 100), "% de puntos como ruido")

ruido = fatales[fatales["foco"] == -1]
en_focos = fatales[fatales["foco"] >= 0]
plt.figure(figsize=(7, 7))
plt.scatter(ruido["lon"], ruido["lat"], s=5, c="lightgray", label="ruido")
plt.scatter(en_focos["lon"], en_focos["lat"], s=14, c=en_focos["foco"], cmap="tab20", label="focos")
plt.legend()
plt.title("DBSCAN: focos de incidentes fatales (300 m)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
savefig("14_dbscan_fatales")

# Aprendizaje SUPERVISADO: le damos al modelo ejemplos con la respuesta conocida
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from limpieza import load_data, savefig, SEED

sns.set_theme(style="white")
df = load_data()

# Variables de entrada (X)
# Voy a usar solo información que se conoce al momento del incidente 
# (NO uso la gravedad como entrada porque es justo lo que queremos predecir)

# Variables nuevas fáciles de entender
df["is_weekend"] = (df["weekday"] >= 5).astype(int)                      # sábado o domingo
df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)  # de 10pm a 5am
df["coord_missing"] = df["lat"].isna().astype(int)                       # 1 si falta la coordenada

# Rellenar los vacíos: coordenadas con la mediana, comuna con "Desconocida"
df["lat"] = df["lat"].fillna(df["lat"].median())
df["lon"] = df["lon"].fillna(df["lon"].median())
df["comuna"] = df["comuna"].fillna("Desconocida")

# Las variables de texto (clase, comuna) se convierten en columnas de 0 y 1
# (una columna por categoría). Esto se llama "one-hot encoding".
texto_a_numeros = pd.get_dummies(df[["clase", "comuna"]], dtype=int)

columnas_numericas = ["hour", "weekday", "month", "is_weekend", "is_night",
                      "lat", "lon", "coord_missing"]
X = pd.concat([df[columnas_numericas], texto_a_numeros], axis=1)
print("Tamaño de X:", X.shape)

# 2) Función que entrena todos los modelos para un objetivo (y) y mide qué tan bien lo hacen
def evaluar_tarea(nombre_tarea, y):
    print()
    print("===== Tarea:", nombre_tarea, "=====")
    print("Porcentaje de casos positivos:", round(y.mean() * 100, 2), "%")

    # Separo 75% para entrenar y 25% para probar con datos que el modelo nunca vio.
    # stratify=y mantiene la misma proporción de positivos en los dos grupos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    # Los modelos que comparo
    modelos = {}
    modelos["Dummy (mayoría)"] = DummyClassifier(strategy="most_frequent")
    modelos["Dummy (estratificado)"] = DummyClassifier(strategy="stratified", random_state=SEED)
    # La regresión logística necesita las variables en la misma escala, por eso
    # uso un Pipeline: primero escala (StandardScaler) y luego entrena el modelo
    modelos["Regresión logística"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, class_weight="balanced"))
    modelos["Árbol de decisión"] = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=SEED)
    modelos["Random Forest"] = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)

    resultados = []          # una fila de métricas por modelo
    matrices = {}            # matriz de confusión de cada modelo
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        print("Entrenando:", nombre)
        modelo.fit(X_train, y_train)

        prediccion = modelo.predict(X_test)                  # 0 o 1
        probabilidad = modelo.predict_proba(X_test)[:, 1]    # probabilidad de ser 1

        fila = {}
        fila["modelo"] = nombre
        fila["accuracy"] = accuracy_score(y_test, prediccion)
        fila["balanced_acc"] = balanced_accuracy_score(y_test, prediccion)
        fila["F1"] = f1_score(y_test, prediccion, zero_division=0)
        fila["ROC-AUC"] = roc_auc_score(y_test, probabilidad)
        fila["PR-AUC"] = average_precision_score(y_test, probabilidad)
        resultados.append(fila)

        matrices[nombre] = confusion_matrix(y_test, prediccion, normalize="true")
        modelos_entrenados[nombre] = modelo

    tabla_resultados = pd.DataFrame(resultados).set_index("modelo").round(3)
    print(tabla_resultados)
    return tabla_resultados, matrices, modelos_entrenados

#Tareas
# 1. ¿hubo heridos o muertos, o solo daños materiales?
y_a = (df["gravedad"] != "SOLO DAÑOS").astype(int)
res_a, mat_a, modelos_a = evaluar_tarea("A - Con víctimas vs. solo daños", y_a)

# 2. ¿hubo muertos? (solo 0.4% de los casos: clases muy desbalanceadas)
y_b = (df["gravedad"] == "MUERTO").astype(int)
res_b, mat_b, modelos_b = evaluar_tarea("B - Fatal vs. no fatal", y_b)

# Mapa de calor: comparación de modelos x métrica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
sns.heatmap(res_a, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar=False, ax=ax1)
ax1.set_title("Tarea A: víctimas vs. solo daños")
ax1.set_ylabel("")
sns.heatmap(res_b, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar=False, ax=ax2)
ax2.set_title("Tarea B: fatal vs. no fatal")
ax2.set_ylabel("")
savefig("06_heatmap_metricas_modelos")

# Mapas de calor: matrices de confusión (cada fila suma 1 = acierto por clase)
for letra, matrices in [("A", mat_a), ("B", mat_b)]:
    fig, ejes = plt.subplots(1, 5, figsize=(18, 3.6))
    i = 0
    for nombre, matriz in matrices.items():
        sns.heatmap(matriz, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, ax=ejes[i], xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
        ejes[i].set_title(nombre, fontsize=10)
        ejes[i].set_xlabel("Predicho")
        ejes[i].set_ylabel("Real")
        i = i + 1
    savefig("07_confusion_tarea_" + letra)

# Mapa de calor: ¿qué variables importan más para el Random Forest? (tarea 1)
bosque = modelos_a["Random Forest"]
importancias = pd.Series(bosque.feature_importances_, index=X.columns)
top15 = importancias.sort_values(ascending=False).head(15)

plt.figure(figsize=(4, 6))
sns.heatmap(top15.to_frame("importancia"), annot=True, fmt=".3f", cmap="Oranges", cbar=False)
plt.title("Top 15 variables (RF, tarea A)")
savefig("08_heatmap_importancia_variables")

# Árbol de decisión pequeño para ver cómo "piensa"
arbol = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=SEED)
arbol.fit(X, y_a)

plt.figure(figsize=(18, 7))
plot_tree(arbol, feature_names=list(X.columns),
          class_names=["Solo daños", "Víctimas"], filled=True, rounded=True, fontsize=8)
savefig("09_arbol_decision")

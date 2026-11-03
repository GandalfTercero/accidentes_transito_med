"""
Funciones que usarán los tres scripts: cargar datos y guardar figuras.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Rutas del proyecto ---

RAIZ = Path(__file__).resolve().parent.parent # Mantener la ruta del proyecto fija
RUTA_EXCEL = RAIZ / "datos" / "Fatal_Road_Traffic.xlsx" #apunta de forma segura al lugar exacto donde están los datos
RUTA_PARQUET = RAIZ / "datos" / "incidentes.parquet"   # ruta de la copia de los datos limpios, para no tener que leer el Excel sin procesar
CARPETA_FIGURAS = RAIZ / "figuras" #Crea la carpeta figuras/ si no existe, para guardar las figuras de los análisis
CARPETA_FIGURAS.mkdir(exist_ok=True)

SEED = 42   # Para que los resultados sean siempre los mismos a la hora de ejecutar el código (aleatoriedad controlada)
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def load_data():
    """Carga los datos limpios. La primera vez lee el Excel (tarda 1-2 min)
    y guarda una copia en parquet; las siguientes veces lee la copia."""
    if RUTA_PARQUET.exists():
        return pd.read_parquet(RUTA_PARQUET)

    print("Limpiando el excel...")
    df = pd.read_excel(RUTA_EXCEL)

    # Cambiar el nombre de algunas columnas
    df = df.rename(columns={
        "AÑO": "year",
        "FECHA_INCIDENTE": "date",
        "CLASE_INCIDENTE": "clase",
        "GRAVEDAD_INCIDENTE": "gravedad",
        "Latitud": "lat",
        "Longitud": "lon",
        "Comuna Planeacion": "comuna",
        "Barrio Planeacion": "barrio",
    })

    # columnas nuevas respecto a fecha y hora
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday   # 0 = lunes, 6 = domingo
    df["month"] = df["date"].dt.month
    
    df["hour"] = df["HORA_INCIDENTE"].astype(str).str[:2].astype(int)

    # 3) Limpiar comuna y barrio: algunas celdas mezclan números con texto,
    #    así que todo se pasa a texto. Los "N/D" y "Sin inf" pasan a valor vacío (NaN)
    df["comuna"] = df["comuna"].astype(str)
    df["barrio"] = df["barrio"].astype(str)
    df["comuna"] = df["comuna"].replace(["N/D", "nan"], np.nan)
    df["barrio"] = df["barrio"].replace(["Sin inf", "nan"], np.nan)


    columnas = ["LLAVE", "date", "year", "month", "weekday", "hour", "clase",
                "gravedad", "lat", "lon", "comuna", "barrio"]
    df = df[columnas]

    df.to_parquet(RUTA_PARQUET, index=False)
    return df


def savefig(nombre):
    """Guarda la figura actual en la carpeta figuras/ y la cierra."""
    plt.tight_layout() #corregir automáticamente los márgenes de la figura
    plt.savefig(CARPETA_FIGURAS / (nombre + ".png"), dpi=130)
    plt.close()
    print("  figura guardada: figuras/" + nombre + ".png")

"""
Funciones que usarán los tres scripts: cargar datos y guardar figuras.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Rutas del proyecto ---
# __file__ es este archivo (src/common.py). parent = src, parent.parent = raíz del proyecto
RAIZ = Path(__file__).resolve().parent.parent
RUTA_EXCEL = RAIZ / "datos" / "Fatal_Road_Traffic.xlsx"
RUTA_PARQUET = RAIZ / "datos" / "incidentes.parquet"   # copia rápida del Excel
CARPETA_FIGURAS = RAIZ / "figuras"
CARPETA_FIGURAS.mkdir(exist_ok=True)

SEED = 42   # semilla para que los resultados sean siempre los mismos
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def load_data():
    """Carga los datos limpios. La primera vez lee el Excel (tarda 1-2 min)
    y guarda una copia en parquet; las siguientes veces lee la copia."""
    if RUTA_PARQUET.exists():
        return pd.read_parquet(RUTA_PARQUET)

    print("Leyendo el Excel (solo la primera vez)...")
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
    plt.tight_layout()
    plt.savefig(CARPETA_FIGURAS / (nombre + ".png"), dpi=130)
    plt.close()
    print("  figura guardada: figuras/" + nombre + ".png")

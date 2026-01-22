import pandas as pd
import numpy as np

# =============================
# Este código integra los datos de los dos sensores 
# del microcontrolador [emg y dinamómetro (código comentado)]    
# con los datos del IMU y los etiqueta con el tiempo en la app,
# y genera un dataset por sesión para después integrarlos en uno solo 
# para el entrenamiento del modelo de machine learning
# =============================
WINDOW_SIZE = 1.0      # segundos
EMG_MIN_SAMPLES = 80   # 80% de 100 Hz
IMU_MIN_SAMPLES = 40   # tolerancia mayor

# =============================
# FEATURES AUXILIARES
# =============================
def compute_zc(signal):
    """Calcula cruces por cero en la señal"""
    return np.sum(np.diff(np.sign(signal)) != 0)

def compute_slope(signal):
    """Calcula la pendiente de la señal usando ajuste lineal"""
    return np.polyfit(np.arange(len(signal)), signal, 1)[0]

# =============================
# PIPELINE PRINCIPAL
# =============================
def integrar_sensores(
    arduino_csv,
    imu_csv,
    sujeto,
    condicion,
    out_features_csv="dataset_features_sesion.csv"
):
    """
    Integra datos de EMG, fuerza e IMU para extraer características.
    
    Parámetros:
    -----------
    arduino_csv : str
        Ruta al archivo CSV con datos de EMG y fuerza del ESP32
        Columnas esperadas: t_app, emg_raw, force_raw
    imu_csv : str
        Ruta al archivo CSV con datos del IMU
        Columnas esperadas: t_app, ax, ay, az
    sujeto : str
        ID del sujeto (ej: "S01", "S02")
    condicion : str
        Condición experimental (ej: "MVC", "30MVC", "50MVC")
    out_features_csv : str
        Ruta de salida para el archivo de características
    
    Retorna:
    --------
    dataset : DataFrame
        Dataset con características extraídas por ventana
    """
    # -------------------------
    # 1. Cargar datos
    # -------------------------
    emg = pd.read_csv(arduino_csv)
    imu = pd.read_csv(imu_csv)

    emg["t_app"] = emg["t_app"].astype(float)
    imu["t_app"] = imu["t_app"].astype(float)

    # -------------------------
    # 2. Tiempo relativo común
    # -------------------------
    t0 = min(emg["t_app"].min(), imu["t_app"].min())
    emg["t_rel"] = emg["t_app"] - t0
    imu["t_rel"] = imu["t_app"] - t0

    emg["window_id"] = (emg["t_rel"] // WINDOW_SIZE).astype(int)
    imu["window_id"] = (imu["t_rel"] // WINDOW_SIZE).astype(int)

    # -------------------------
    # 3. EMG + FUERZA
    # -------------------------
    emg_features = []

    for wid, g in emg.groupby("window_id"):
        if len(g) < EMG_MIN_SAMPLES:
            continue

        emg_sig = g["emg_raw"].values
        force_sig = g["force_raw"].values if "force_raw" in g.columns else np.zeros(len(g))

        emg_features.append({
            "window_id": wid,
            "emg_rms": np.sqrt(np.mean(emg_sig**2)),
            "emg_mean_abs": np.mean(np.abs(emg_sig)),
            "emg_var": np.var(emg_sig),
            "emg_max": np.max(emg_sig),
            "emg_zc": compute_zc(emg_sig),
            "force_mean": np.mean(force_sig),
            "force_max": np.max(force_sig),
            "force_slope": compute_slope(force_sig),
            "emg_samples": len(emg_sig)
        })

    emg_df = pd.DataFrame(emg_features)

    # -------------------------
    # 4. IMU
    # -------------------------
    imu_features = []

    for wid, g in imu.groupby("window_id"):
        if len(g) < IMU_MIN_SAMPLES:
            continue

        imu_features.append({
            "window_id": wid,
            "ax_mean": g["ax"].mean(),
            "ax_var": g["ax"].var(),
            "ay_mean": g["ay"].mean(),
            "ay_var": g["ay"].var(),
            "az_mean": g["az"].mean(),
            "az_var": g["az"].var(),
            "acc_magnitude_mean": np.mean(
                np.sqrt(g["ax"]**2 + g["ay"]**2 + g["az"]**2)
            ),
            "imu_samples": len(g)
        })

    imu_df = pd.DataFrame(imu_features)

    # -------------------------
    # 5. FUSIÓN FINAL
    # -------------------------
    dataset = pd.merge(
        emg_df,
        imu_df,
        on="window_id",
        how="inner"
    )

    # -------------------------
    # 6. AGREGAR METADATOS
    # -------------------------
    dataset["sujeto"] = sujeto
    dataset["condicion_contraccion"] = condicion

    # Reordenar columnas
    cols = ["sujeto", "condicion_contraccion", "window_id"] + \
           [c for c in dataset.columns if c not in
            ["sujeto", "condicion_contraccion", "window_id"]]

    dataset = dataset[cols]

    # -------------------------
    # 7. GUARDAR
    # -------------------------
    dataset.to_csv(out_features_csv, index=False)
    print(f"Dataset de la sesión generado: {out_features_csv}")
    print(f"Total de ventanas: {len(dataset)}")
    print(dataset.head())

    return dataset


if __name__ == "__main__":
    # Ejemplo de uso
    print("Script de integración de sensores")
    print("Uso:")
    print("  from build_dataset import integrar_sensores")
    print("  integrar_sensores(arduino_csv, imu_csv, sujeto, condicion, out_csv)")

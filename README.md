# Scripts de Procesamiento de Datos

## Descripción General
Scripts de utilidad para procesar datos multi-sensor y construir datasets de ML.

## Scripts

### build_dataset.py
**Propósito**: Integra datos de EMG, fuerza e IMU para extraer características para machine learning.

**Uso**:
```python
from build_dataset import integrar_sensores

integrar_sensores(
    arduino_csv="03_data_raw/S01/MVC/emg_force_raw.csv",
    imu_csv="03_data_raw/S01/MVC/imu_raw.csv",
    sujeto="S01",
    condicion="MVC",
    out_features_csv="05_features/ejecuciones/S01_MVC_features.csv"
)
```

**Formato de CSV de Entrada**:
- **arduino_csv**: `t_app`, `emg_raw`, `force_raw`
- **imu_csv**: `t_app`, `ax`, `ay`, `az`

**Características de Salida**:
- EMG: RMS, Media Abs, Varianza, Máximo, Cruces por Cero
- Fuerza: Media, Máximo, Pendiente
- IMU: Media/varianza de aceleración por eje, magnitud

**Parámetros**:
- `WINDOW_SIZE`: 1.0 segundo (configurable)
- `EMG_MIN_SAMPLES`: 80 muestras mínimo (80% de 100 Hz)
- `IMU_MIN_SAMPLES`: 40 muestras mínimo

---

### append_dataset.py
**Propósito**: Fusiona datasets de sesiones individuales en un dataset global de ML.

**Uso**:
```python
from append_dataset import append_a_dataset_global

append_a_dataset_global(
    csv_ejecucion="05_features/ejecuciones/S01_MVC_features.csv",
    csv_global="05_features/dataset_features_ml.csv"
)
```

**Características**:
- Crea el dataset global si no existe
- Valida consistencia de columnas
- Previene desajustes de esquema
- Rastrea el crecimiento del dataset

---

## Ejemplo de Flujo de Trabajo

```python
# Paso 1: Procesar cada sesión
integrar_sensores(
    "03_data_raw/S01/MVC/emg_force_raw.csv",
    "03_data_raw/S01/MVC/imu_raw.csv",
    "S01", "MVC",
    "05_features/ejecuciones/S01_MVC_features.csv"
)

# Paso 2: Agregar al dataset global
append_a_dataset_global(
    "05_features/ejecuciones/S01_MVC_features.csv",
    "05_features/dataset_features_ml.csv"
)

# Repetir para todos los sujetos y condiciones
```

## Dependencias
```bash
pip install pandas numpy
```

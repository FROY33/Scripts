import pandas as pd
import os

def append_a_dataset_global(
    csv_ejecucion,
    csv_global="dataset_features_ml.csv"
):
    """
    Agrega un CSV de una ejecución al dataset global de ML,
    validando estructura y columnas.
    """

    # -------------------------
    # 1. Cargar CSV de ejecución
    # -------------------------
    df_new = pd.read_csv(csv_ejecucion)

    # -------------------------
    # 2. Si el global no existe, crear
    # -------------------------
    if not os.path.exists(csv_global):
        df_new.to_csv(csv_global, index=False)
        print(f"Dataset global creado: {csv_global}")
        return

    # -------------------------
    # 3. Cargar dataset global
    # -------------------------
    df_global = pd.read_csv(csv_global)

    # -------------------------
    # 4. Validar columnas
    # -------------------------
    if set(df_new.columns) != set(df_global.columns):
        raise ValueError(
            "Las columnas no coinciden entre el CSV de ejecución y el dataset global.\n"
            f"Ejecución: {set(df_new.columns)}\n"
            f"Global: {set(df_global.columns)}"
        )

    # -------------------------
    # 5. Reordenar columnas (por seguridad)
    # -------------------------
    df_new = df_new[df_global.columns]

    # -------------------------
    # 6. Append
    # -------------------------
    df_final = pd.concat([df_global, df_new], ignore_index=True)

    # -------------------------
    # 7. Guardar
    # -------------------------
    df_final.to_csv(csv_global, index=False)

    print(f"Ejecución agregada correctamente a {csv_global}")
    print(f"Filas nuevas: {len(df_new)} | Total: {len(df_final)}")

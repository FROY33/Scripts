import pandas as pd
import os

def append_a_dataset_global(
    csv_ejecucion,
    csv_global="dataset_features_ml.csv"
):
    """
    Agrega un CSV de una ejecución al dataset global de ML,
    validando estructura y columnas.
    
    Parámetros:
    -----------
    csv_ejecucion : str
        Ruta al archivo CSV de una sesión individual
    csv_global : str
        Ruta al archivo CSV del dataset global (se crea si no existe)
    
    Funcionalidad:
    --------------
    - Crea el dataset global si no existe
    - Valida que las columnas coincidan
    - Agrega los datos nuevos al final
    - Reporta el crecimiento del dataset
    """

    # -------------------------
    # 1. Cargar CSV de ejecución
    # -------------------------
    df_new = pd.read_csv(csv_ejecucion)
    print(f"Cargando sesión: {csv_ejecucion}")
    print(f"  Filas: {len(df_new)}")
    print(f"  Columnas: {len(df_new.columns)}")

    # -------------------------
    # 2. Si el global no existe, crear
    # -------------------------
    if not os.path.exists(csv_global):
        df_new.to_csv(csv_global, index=False)
        print(f"\n✅ Dataset global creado: {csv_global}")
        print(f"  Total de filas: {len(df_new)}")
        return

    # -------------------------
    # 3. Cargar dataset global
    # -------------------------
    df_global = pd.read_csv(csv_global)
    print(f"\nDataset global existente: {csv_global}")
    print(f"  Filas actuales: {len(df_global)}")

    # -------------------------
    # 4. Validar columnas
    # -------------------------
    if set(df_new.columns) != set(df_global.columns):
        raise ValueError(
            "❌ Las columnas no coinciden entre el CSV de ejecución y el dataset global.\n"
            f"Ejecución: {set(df_new.columns)}\n"
            f"Global: {set(df_global.columns)}\n"
            f"Diferencia: {set(df_new.columns).symmetric_difference(set(df_global.columns))}"
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

    print(f"\n✅ Ejecución agregada correctamente a {csv_global}")
    print(f"  Filas nuevas: {len(df_new)}")
    print(f"  Total de filas: {len(df_final)}")
    print(f"  Crecimiento: +{len(df_new)/len(df_global)*100:.1f}%")


if __name__ == "__main__":
    # Ejemplo de uso
    print("Script de agregación de datasets")
    print("Uso:")
    print("  from append_dataset import append_a_dataset_global")
    print("  append_a_dataset_global(csv_ejecucion, csv_global)")

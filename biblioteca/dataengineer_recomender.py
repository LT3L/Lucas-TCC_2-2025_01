import pandas as pd
import psutil
import platform
import joblib
import os
from pathlib import Path

# Carregamento dos modelos salvos
MODELO_PATH = "../datasets_and_models_output/modelo_recomendador.pkl"
ENCODER_PATH = "../datasets_and_models_output/label_encoder.pkl"
FEATURES_PATH = "../datasets_and_models_output/modelo_features.pkl"
feature_order = joblib.load(FEATURES_PATH)

modelo = joblib.load(MODELO_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def coletar_info_sistema():
    return {
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "frequencia_cpu_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "memoria_total_mb": psutil.virtual_memory().total / (1024 ** 2),
        "disco_total_gb": psutil.disk_usage('/').total / (1024 ** 3),
    }

def analisar_amostra(path, nrows=5000):
    ext = Path(path).suffix.lower().replace('.', '')
    if ext == 'csv':
        df = pd.read_csv(path, nrows=nrows)
    elif ext == 'parquet':
        df = pd.read_parquet(path)
    elif ext == 'json':
        df = pd.read_json(path, lines=True, nrows=nrows)
    else:
        raise ValueError("Formato de arquivo não suportado.")

    tipos = df.dtypes
    total = len(tipos)
    return {
        "num_linhas": len(df),
        "num_colunas": df.shape[1],
        "percentual_numerico": tipos.isin(['int64', 'float64']).mean(),
        "percentual_string": (tipos == 'object').mean(),
        "percentual_datetime": (tipos == 'datetime64[ns]').mean(),
    }

def recomendar_biblioteca(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Arquivo de dataset não encontrado.")

    info_sistema = coletar_info_sistema()

    print("\nConfigurações do sistema detectadas:")
    for k, v in info_sistema.items():
        print(f"  - {k.replace('_', ' ').capitalize()}: {round(v, 2) if isinstance(v, float) else v}")

    info_dados = analisar_amostra(dataset_path)

    print("\nPerfil do dataset analisado:")
    print(f"  - Arquivo: {dataset_path}")
    for k, v in info_dados.items():
        print(f"  - {k.replace('_', ' ').capitalize()}: {round(v, 4) if isinstance(v, float) else v}")

    tamanho_dataset_bytes = os.path.getsize(dataset_path)
    tamanho_dataset_nominal_mb = tamanho_dataset_bytes / (1024 * 1024)

    entrada = {
        "tamanho_dataset_nominal_mb": tamanho_dataset_nominal_mb,
        "cpu_medio_execucao": 50.0,  # Placeholder
        "memoria_media_execucao": 1000.0,  # Placeholder
        "leitura_bytes": tamanho_dataset_bytes,
        "escrita_bytes": 0,  # Placeholder
        "tamanho_dataset_bytes": tamanho_dataset_bytes,
        **info_sistema,
        **info_dados
    }

    df_entrada = pd.DataFrame([entrada])
    df_entrada = df_entrada[feature_order]
    pred = modelo.predict(df_entrada)[0]
    biblioteca_recomendada = label_encoder.inverse_transform([pred])[0]

    print(f"\nBiblioteca recomendada: {biblioteca_recomendada.upper()}")

    return biblioteca_recomendada

if __name__ == "__main__":
    recomendar_biblioteca("/Users/lucas.lima/Documents/Projects/TCC_2/datasets/nyc_taxi/csv/amostra_10MB.csv")
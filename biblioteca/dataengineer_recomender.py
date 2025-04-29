import pandas as pd
import psutil
import platform
import joblib
import os
from pathlib import Path

# Carregamento dos modelos salvos
MODELO_PATH = "/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/modelo_regressao_tempo_execucao.pkl"
FEATURES_PATH = "/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/modelo_regressao_features.pkl"
feature_order = joblib.load(FEATURES_PATH)

modelo = joblib.load(MODELO_PATH)

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

    numericos = tipos.apply(lambda t: pd.api.types.is_numeric_dtype(t)).sum()
    strings = tipos.apply(lambda t: pd.api.types.is_string_dtype(t)).sum()
    datetimes = tipos.apply(lambda t: pd.api.types.is_datetime64_any_dtype(t)).sum()

    return {
        "num_linhas": len(df),
        "num_colunas": df.shape[1],
        "percentual_numerico": numericos / total,
        "percentual_string": strings / total,
        "percentual_datetime": datetimes / total,
    }

def recomendar_biblioteca(dataset_path, tem_joins=False, tem_groupby=False):
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

    resultados = {}
    for biblioteca in ["pandas", "polars", "spark", "duckdb"]:
        entrada = {
            "biblioteca": biblioteca,
            "dataset_formato": Path(dataset_path).suffix.replace('.', ''),
            "tamanho_dataset_nominal_mb": tamanho_dataset_nominal_mb,
            "cpu_medio_execucao": 50.0,
            "memoria_media_execucao": 1000.0,
            "leitura_bytes": tamanho_dataset_bytes,
            "escrita_bytes": 0,
            "tamanho_dataset_bytes": tamanho_dataset_bytes,
            "tem_joins": tem_joins,
            "tem_groupby": tem_groupby,
            **info_sistema,
            **info_dados
        }

        df_entrada = pd.DataFrame([entrada])
        df_entrada = df_entrada[feature_order]
        tempo_previsto = modelo.predict(df_entrada)[0]
        resultados[biblioteca] = tempo_previsto

    melhor_biblioteca = min(resultados, key=resultados.get)
    print(f"\nTempo previsto por biblioteca:")
    for lib, tempo in resultados.items():
        print(f"  - {lib}: {tempo:.2f} segundos")

    print(f"\n📌 Biblioteca recomendada: {melhor_biblioteca.upper()}")
    return melhor_biblioteca

if __name__ == "__main__":
    recomendar_biblioteca("/Users/lucas.lima/Documents/Projects/TCC_2/app/datasets/nyc_taxi/json/amostra_1000MB.json")

    recomendar_biblioteca("/Users/lucas.lima/Documents/Projects/TCC_2/app/datasets/nyc_taxi/json/amostra_10MB.json")

    recomendar_biblioteca("/Users/lucas.lima/Documents/Projects/TCC_2/app/datasets/nyc_taxi/parquet/amostra_100MB.parquet")
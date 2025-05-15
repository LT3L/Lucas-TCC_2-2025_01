"""
Script para testar o modelo de recomendação de tempo de execução.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os
import traceback

# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
MODELS_DIR = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
)
FEATURES_PATH = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/"
    "modelo_features.pkl"
)

def load_model() -> Tuple[object, list, Dict]:
    """Carrega o modelo treinado, lista de features e métricas."""
    regressor = joblib.load(os.path.join(MODELS_DIR, "modelo_regressor.pkl"))
    features = joblib.load(FEATURES_PATH)
    metrics = joblib.load(os.path.join(MODELS_DIR, "metricas_modelos.pkl"))
    return regressor, features, metrics

def prepare_input_data(
    biblioteca: str,
    dataset_formato: str,
    tamanho_mb: float,
    num_linhas: int,
    num_colunas: int,
    percentual_numerico: float,
    percentual_string: float,
    percentual_datetime: float,
    nucleos_fisicos: int,
    nucleos_logicos: int,
    frequencia_cpu_max: float,
    memoria_total_mb: float,
    disco_total_gb: float,
    tem_joins: bool,
    tem_groupby: bool,
    cpu_medio_execucao: float = 0.0,
    memoria_media_execucao: float = 0.0,
    leitura_bytes: int = 0,
    escrita_bytes: int = 0
) -> pd.DataFrame:
    """Prepara os dados de entrada para predição."""
    
    data = pd.DataFrame({
        "biblioteca": [biblioteca],
        "dataset_formato": [dataset_formato],
        "tamanho_dataset_nominal_mb": [tamanho_mb],
        "tamanho_dataset_bytes": [tamanho_mb * 1024 * 1024],
        "num_linhas": [num_linhas],
        "num_colunas": [num_colunas],
        "percentual_numerico": [percentual_numerico],
        "percentual_string": [percentual_string],
        "percentual_datetime": [percentual_datetime],
        "nucleos_fisicos": [nucleos_fisicos],
        "nucleos_logicos": [nucleos_logicos],
        "frequencia_cpu_max": [frequencia_cpu_max],
        "memoria_total_mb": [memoria_total_mb],
        "disco_total_gb": [disco_total_gb],
        "tem_joins": [int(tem_joins)],
        "tem_groupby": [int(tem_groupby)],
        "cpu_medio_execucao": [cpu_medio_execucao],
        "memoria_media_execucao": [memoria_media_execucao],
        "leitura_bytes": [leitura_bytes],
        "escrita_bytes": [escrita_bytes]
    })

    # Features de interação
    data['tamanho_por_linha'] = data['tamanho_dataset_nominal_mb'] / data['num_linhas']
    data['linhas_por_coluna'] = data['num_linhas'] / data['num_colunas']
    data['complexidade_operacao'] = data['tem_joins'] + data['tem_groupby']

    # Log transformações
    numeric_cols = [
        'tamanho_dataset_nominal_mb',
        'tamanho_dataset_bytes',
        'num_linhas',
        'num_colunas',
        'leitura_bytes',
        'escrita_bytes'
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[f'{col}_log'] = np.log1p(data[col])

    return data

def predict_execution_time(
    regressor: object,
    features: list,
    input_data: pd.DataFrame,
    metrics: Dict
) -> Dict:
    """
    Faz predição do tempo de execução usando o modelo de regressão.
    """
    # Garantir que todas as features necessárias estejam presentes
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    X = input_data[features]
    
    # Fazer predição (o modelo retorna log do tempo)
    tempo_estimado_log = regressor.predict(X)[0]
    tempo_estimado = np.expm1(tempo_estimado_log)
    
    # Garantir tempo mínimo de 0.1 segundos
    tempo_estimado = max(0.1, tempo_estimado)
    
    # Calcular intervalo de confiança baseado nas métricas do modelo
    mae = metrics['regressor']['metrics']['mae_mean']
    tempo_min = max(0.1, tempo_estimado - mae)
    tempo_max = tempo_estimado + mae
    
    return {
        "tempo_estimado": tempo_estimado,
        "intervalo_confianca": {
            "min": tempo_min,
            "max": tempo_max
        },
        "biblioteca": input_data['biblioteca'].iloc[0],
        "tamanho_dataset_mb": input_data['tamanho_dataset_nominal_mb'].iloc[0],
        "complexidade": input_data['complexidade_operacao'].iloc[0]
    }

def print_prediction_results(result: Dict) -> None:
    """Exibe os resultados da predição de forma formatada."""
    print("\nResultados da Predição:")
    print("-" * 50)
    print(f"Biblioteca: {result['biblioteca']}")
    print(f"Tamanho do Dataset: {result['tamanho_dataset_mb']:.0f} MB")
    print(f"Complexidade da Operação: {result['complexidade']}")
    print(f"Tempo Estimado: {result['tempo_estimado']:.2f} segundos")
    print(f"Intervalo de Confiança: {result['intervalo_confianca']['min']:.2f} - {result['intervalo_confianca']['max']:.2f} segundos")
    print("-" * 50)

def main():
    """Função principal para demonstrar o uso do modelo."""
    try:
        print("Carregando modelo...")
        regressor, features, metrics = load_model()
        print("Modelo carregado com sucesso!")

        # Exemplo 1: Operação pequena com DuckDB
        print("\nExemplo 1: Operação pequena com DuckDB")
        input_data = prepare_input_data(
            biblioteca="duckdb",
            dataset_formato="parquet",
            tamanho_mb=1,
            num_linhas=10000,
            num_colunas=5,
            percentual_numerico=0.7,
            percentual_string=0.2,
            percentual_datetime=0.1,
            nucleos_fisicos=4,
            nucleos_logicos=8,
            frequencia_cpu_max=3.6,
            memoria_total_mb=164000,
            disco_total_gb=512,
            tem_joins=True,
            tem_groupby=True
        )

        result = predict_execution_time(regressor, features, input_data, metrics)
        print_prediction_results(result)

        # Exemplo 2: Operação grande com Pandas
        print("\nExemplo 2: Operação grande com Pandas")
        input_data = prepare_input_data(
            biblioteca="pandas",
            dataset_formato="parquet",
            tamanho_mb=20000,
            num_linhas=10000000,
            num_colunas=20,
            percentual_numerico=0.5,
            percentual_string=0.3,
            percentual_datetime=0.2,
            nucleos_fisicos=4,
            nucleos_logicos=8,
            frequencia_cpu_max=3.6,
            memoria_total_mb=16384,
            disco_total_gb=512,
            tem_joins=False,
            tem_groupby=True
        )

        result = predict_execution_time(regressor, features, input_data, metrics)
        print_prediction_results(result)

        # Exemplo 3: Comparação entre bibliotecas
        print("\nExemplo 3: Comparação entre Bibliotecas (Dataset de 4000 MB)")
        for biblioteca in ["pandas", "polars", "duckdb"]:
            input_data = prepare_input_data(
                biblioteca=biblioteca,
                dataset_formato="parquet",
                tamanho_mb=4000,
                num_linhas=1000000,
                num_colunas=10,
                percentual_numerico=0.6,
                percentual_string=0.3,
                percentual_datetime=0.1,
                nucleos_fisicos=4,
                nucleos_logicos=8,
                frequencia_cpu_max=3.6,
                memoria_total_mb=16384,
                disco_total_gb=512,
                tem_joins=True,
                tem_groupby=True
            )
            
            result = predict_execution_time(regressor, features, input_data, metrics)
            print_prediction_results(result)

    except Exception as e:
        print("Erro ao executar o script:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

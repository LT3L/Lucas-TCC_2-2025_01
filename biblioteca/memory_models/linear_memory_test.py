"""
Script para testar o modelo linear de classificação de falhas baseado em memória.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import traceback
import warnings

# Suprimir warnings específicos
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
MODELS_DIR = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
)
MEMORY_FEATURES_PATH = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/"
    "linear_memory_features.pkl"
)

def load_linear_model() -> Tuple[object, list, Dict, float]:
    """Carrega o modelo linear de classificação baseado em memória, features e threshold."""
    linear_classifier = joblib.load(os.path.join(MODELS_DIR, "linear_memory_classifier.pkl"))
    memory_features = joblib.load(MEMORY_FEATURES_PATH)
    memory_metrics = joblib.load(os.path.join(MODELS_DIR, "linear_memory_metrics.pkl"))
    memory_threshold = joblib.load(os.path.join(MODELS_DIR, "linear_memory_threshold.pkl"))
    return linear_classifier, memory_features, memory_metrics, memory_threshold

def prepare_memory_input(
    biblioteca: str,
    dataset_formato: str,
    tamanho_mb: float,
    memoria_total_mb: float
) -> pd.DataFrame:
    """Prepara os dados de entrada focados em memória para predição."""
    
    # Criar dataframe base com os dados principais
    data = pd.DataFrame({
        "biblioteca": [biblioteca],
        "dataset_formato": [dataset_formato],
        "tamanho_dataset_nominal_mb": [tamanho_mb],
        "memoria_total_mb": [memoria_total_mb]
    })
    
    # Calcular features de memória
    data['proporcao_memoria'] = data['tamanho_dataset_nominal_mb'] / data['memoria_total_mb']
    
    return data

def predict_memory_failure(
    classifier: object,
    features: List[str],
    input_data: pd.DataFrame,
    threshold: float = 0.4
) -> Dict:
    """
    Faz predições sobre falhas de memória usando o modelo linear.
    
    Retorna:
        Dict com as predições e métricas
    """
    # Garantir que todas as features necessárias estão presentes
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    
    # Reordenar colunas para corresponder à ordem esperada
    X = input_data[features].copy()
    X.columns = features
    
    # Predizer probabilidade de sucesso
    success_prob = classifier.predict_proba(X)[0][1]
    
    # Estruturar resultado
    result = {
        "probabilidade_sucesso": success_prob,
        "vai_falhar": success_prob < threshold,
        "proporcao_memoria": input_data['proporcao_memoria'].iloc[0],
        "memoria_disponivel_mb": input_data['memoria_total_mb'].iloc[0],
        "tamanho_dataset_mb": input_data['tamanho_dataset_nominal_mb'].iloc[0],
        "biblioteca": input_data['biblioteca'].iloc[0],
        "fator_risco": 1 - success_prob
    }
    
    return result

def print_prediction_results(result: Dict):
    """Exibe os resultados da predição de forma formatada."""
    try:
        print("\nResultados da predição:")
        print(f"Biblioteca: {result['biblioteca']}")
        print(f"Tamanho do dataset: {result['tamanho_dataset_mb']:.1f} MB")
        print(f"Memória disponível: {result['memoria_disponivel_mb']:.1f} MB")
        print(f"Proporção de memória: {result['proporcao_memoria']:.2%}")
        print(f"Probabilidade de sucesso: {result['probabilidade_sucesso']:.2%}")
        print(f"Fator de risco: {result['fator_risco']:.2f} (0-1)")
        print(f"Vai falhar: {'Sim' if result['vai_falhar'] else 'Não'}")
        print()
    except Exception as e:
        print(f"Erro ao exibir resultados: {str(e)}")

def main():
    """Função principal para demonstrar o uso do classificador linear."""
    try:
        print("Carregando modelo linear de classificação de memória...")
        linear_classifier, memory_features, memory_metrics, memory_threshold = load_linear_model()
        print("Modelo carregado com sucesso!")
        
        print("\n===== TESTE DO CLASSIFICADOR LINEAR DE MEMÓRIA =====")
        
        # Exemplo 1: Dataset pequeno
        print("\nExemplo 1: Dataset pequeno em DuckDB")
        input_data = prepare_memory_input(
            biblioteca="duckdb",
            dataset_formato="parquet",
            tamanho_mb=10,
            memoria_total_mb=16384  # 16GB
        )
        
        result = predict_memory_failure(linear_classifier, memory_features, input_data, memory_threshold)
        print_prediction_results(result)
        
        # Exemplo 2: Dataset grande
        print("Exemplo 2: Dataset grande em Pandas")
        input_data = prepare_memory_input(
            biblioteca="pandas",
            dataset_formato="csv", 
            tamanho_mb=1000,  # 8GB
            memoria_total_mb=16384  # 16GB
        )
        
        result = predict_memory_failure(linear_classifier, memory_features, input_data, memory_threshold)
        print_prediction_results(result)
        
        # Exemplo 3: Caso limítrofe
        print("Exemplo 3: Caso limítrofe em Polars")
        input_data = prepare_memory_input(
            biblioteca="polars",
            dataset_formato="parquet",
            tamanho_mb=5000,  # 5GB
            memoria_total_mb=16384  # 16GB
        )
        
        result = predict_memory_failure(linear_classifier, memory_features, input_data, memory_threshold)
        print_prediction_results(result)
        
        # Exemplo 4: Dataset muito grande
        print("Exemplo 4: Dataset muito grande")
        input_data = prepare_memory_input(
            biblioteca="duckdb",
            dataset_formato="parquet",
            tamanho_mb=20000,  # 20GB
            memoria_total_mb=16384  # 16GB
        )
        
        result = predict_memory_failure(linear_classifier, memory_features, input_data, memory_threshold)
        print_prediction_results(result)
        
        # Exemplo 5: Comparação entre bibliotecas
        print("===== COMPARAÇÃO ENTRE BIBLIOTECAS =====")
        dataset_size = 4000  # 4GB
        libraries = ["pandas", "polars", "duckdb"]
        
        print(f"Dataset de {dataset_size} MB, formato parquet:")
        for lib in libraries:
            input_data = prepare_memory_input(
                biblioteca=lib,
                dataset_formato="parquet",
                tamanho_mb=dataset_size,
                memoria_total_mb=16384  # 16GB
            )
            
            result = predict_memory_failure(linear_classifier, memory_features, input_data, memory_threshold)
            print(f"- {lib.upper()}:")
            print(f"  Probabilidade de sucesso: {result['probabilidade_sucesso']:.2%}")
            print(f"  Vai falhar: {'Sim' if result['vai_falhar'] else 'Não'}")
            print(f"  Fator de risco: {result['fator_risco']:.2f}")
            print()
        
    except Exception as e:
        print("Erro ao executar o script:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 
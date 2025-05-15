"""
Testa o modelo de Random Forest para predição de falhas baseado em uso de memória.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os

# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
MODELS_DIR = "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
MEMORY_FEATURES_PATH = "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/forest_memory_features.pkl"

def load_forest_model() -> tuple:
    """Carrega o modelo Random Forest e seus componentes."""
    forest_classifier = joblib.load(os.path.join(MODELS_DIR, "forest_memory_classifier.pkl"))
    memory_features = joblib.load(MEMORY_FEATURES_PATH)
    metrics = joblib.load(os.path.join(MODELS_DIR, "forest_memory_metrics.pkl"))
    threshold = joblib.load(os.path.join(MODELS_DIR, "forest_memory_threshold.pkl"))
    
    return forest_classifier, memory_features, metrics, threshold

def prepare_memory_input(
    dataset_size_mb: float,
    total_memory_mb: float,
    biblioteca: str,
    formato: str
) -> pd.DataFrame:
    """Prepara os dados de entrada para predição."""
    # Criar DataFrame com as features necessárias
    data = {
        'tamanho_dataset_nominal_mb': [dataset_size_mb],
        'memoria_total_mb': [total_memory_mb],
        'biblioteca': [biblioteca],
        'dataset_formato': [formato]
    }
    
    df = pd.DataFrame(data)
    
    # Calcular proporção de memória
    df['proporcao_memoria'] = df['tamanho_dataset_nominal_mb'] / df['memoria_total_mb']
    
    return df

def predict_memory_failure(
    dataset_size_mb: float,
    total_memory_mb: float,
    biblioteca: str,
    formato: str
) -> Dict[str, Any]:
    """Faz a predição de falha baseada em memória usando o modelo Random Forest."""
    try:
        # Carregar modelo e componentes
        forest_classifier, memory_features, metrics, threshold = load_forest_model()
        
        # Preparar dados de entrada
        input_data = prepare_memory_input(dataset_size_mb, total_memory_mb, biblioteca, formato)
        
        # Fazer predição
        success_prob = forest_classifier.predict_proba(input_data)[0, 1]
        will_fail = success_prob < threshold
        
        return {
            'success_probability': success_prob,
            'will_fail': will_fail,
            'memory_proportion': input_data['proporcao_memoria'].iloc[0],
            'available_memory_mb': total_memory_mb,
            'dataset_size_mb': dataset_size_mb,
            'risk_factor': 1 - success_prob
        }
        
    except Exception as e:
        print(f"Erro ao fazer predição: {str(e)}")
        return None

def print_prediction_results(results: Optional[Dict[str, Any]]) -> None:
    """Exibe os resultados da predição de forma formatada."""
    if results is None:
        print("Não foi possível fazer a predição.")
        return
    
    print("\nResultados da Predição:")
    print("-" * 50)
    print(f"Probabilidade de Sucesso: {results['success_probability']:.1%}")
    print(f"Indicação de Falha: {'Sim' if results['will_fail'] else 'Não'}")
    print(f"Proporção de Memória: {results['memory_proportion']:.1%}")
    print(f"Memória Disponível: {results['available_memory_mb']:.0f} MB")
    print(f"Tamanho do Dataset: {results['dataset_size_mb']:.0f} MB")
    print(f"Fator de Risco: {results['risk_factor']:.1%}")
    print("-" * 50)

def main():
    """Função principal para demonstrar o uso do classificador."""
    # Exemplo 1: Dataset pequeno
    print("\nTeste 1: Dataset Pequeno (DuckDB) - parquet")
    results = predict_memory_failure(
        dataset_size_mb=100,
        total_memory_mb=8000,
        biblioteca="duckdb",
        formato="parquet"
    )
    print_prediction_results(results)
    
    # Exemplo 2: Dataset grande
    print("\nTeste 2: Dataset Grande (Pandas) - csv")
    results = predict_memory_failure(
        dataset_size_mb=4000,
        total_memory_mb=8000,
        biblioteca="pandas",
        formato="csv"
    )
    print_prediction_results(results)
    
    # Exemplo 3: Caso limite
    print("\nTeste 3: Caso Limite (Polars) - parquet")
    results = predict_memory_failure(
        dataset_size_mb=2000,
        total_memory_mb=4000,
        biblioteca="polars",
        formato="parquet"
    )
    print_prediction_results(results)
    
    # Exemplo 4: Comparação entre bibliotecas
    print("\nTeste 4: Comparação entre Bibliotecas (Dataset de 4000 MB - parquet)")
    for biblioteca in ["pandas", "polars", "duckdb"]:
        print(f"\nBiblioteca: {biblioteca}")
        results = predict_memory_failure(
            dataset_size_mb=4000,
            total_memory_mb=8000,
            biblioteca=biblioteca,
            formato="parquet"
        )
        print_prediction_results(results)

if __name__ == "__main__":
    main() 
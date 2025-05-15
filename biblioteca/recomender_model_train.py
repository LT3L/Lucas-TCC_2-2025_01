"""
Treina e valida modelos de regressão para estimativa de tempo de execução.
Usa múltiplos modelos para estimativa de tempo, considerando apenas execuções bem-sucedidas.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import os

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
CSV_PATH = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/benchmark_WIN-KIOBB81FP3L_20250512_173438.csv"
)
MODELS_OUT_DIR = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
)
FEATURES_OUT = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/"
    "modelo_features.pkl"
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa os dados antes do treinamento."""
    df = df.copy()
    
    # Criar features de interação
    if 'tamanho_dataset_nominal_mb' in df.columns and 'num_linhas' in df.columns:
        df['tamanho_por_linha'] = df['tamanho_dataset_nominal_mb'] / df['num_linhas']
    
    if 'num_linhas' in df.columns and 'num_colunas' in df.columns:
        df['linhas_por_coluna'] = df['num_linhas'] / df['num_colunas']
    
    # Criar features de complexidade
    df['complexidade_operacao'] = df['tem_joins'].astype(int) + df['tem_groupby'].astype(int)
    
    # Normalizar percentuais para somarem 1
    percent_cols = ['percentual_numerico', 'percentual_string', 'percentual_datetime']
    if all(col in df.columns for col in percent_cols):
        total = df[percent_cols].sum(axis=1)
        for col in percent_cols:
            df[col] = df[col] / total
    
    # Log transform para variáveis numéricas com distribuição assimétrica
    numeric_cols = [
        'tamanho_dataset_nominal_mb',
        'tamanho_dataset_bytes',
        'num_linhas',
        'num_colunas',
        'leitura_bytes',
        'escrita_bytes'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    return df

def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Carrega e prepara o dataset para treinamento."""
    df = pd.read_csv(path)

    if "dataset_id" not in df.columns:
        raise KeyError("Coluna 'dataset_id' não encontrada no CSV — necessária para GroupKFold.")
    
    # Filtrar apenas execuções bem-sucedidas
    df = df[df['status'] == 'completed'].copy()
    
    # Verificar e limpar dados
    print("\nVerificando dados de entrada:")
    print(f"Total de registros (apenas sucessos): {len(df)}")
    
    # Remover registros com valores nulos em features importantes
    important_cols = ['tempo_execucao', 'biblioteca', 'dataset_formato', 
                     'tamanho_dataset_nominal_mb']
    df = df.dropna(subset=important_cols)
    print(f"Registros após remoção de nulos: {len(df)}")
    
    # Pré-processar dados
    df = preprocess_data(df)
    
    # Preparar features
    categorical_features = [
        "biblioteca", 
        "dataset_formato"
    ]
    
    numeric_features = [
        "tamanho_dataset_nominal_mb",
        "cpu_medio_execucao",
        "memoria_media_execucao",
        "leitura_bytes",
        "escrita_bytes",
        "nucleos_fisicos",
        "nucleos_logicos",
        "frequencia_cpu_max",
        "memoria_total_mb",
        "disco_total_gb",
        "tem_joins",
        "tem_groupby"
    ]
    
    # Adicionar features log transformadas
    log_features = [f'{col}_log' for col in numeric_features if f'{col}_log' in df.columns]
    numeric_features.extend(log_features)

    # Verificar features disponíveis
    available = [c for c in categorical_features + numeric_features if c in df.columns]
    print(f"\nFeatures disponíveis: {len(available)}")
    print(f"Features categóricas: {[c for c in categorical_features if c in available]}")
    print(f"Features numéricas: {[c for c in numeric_features if c in available]}")
    
    X = df[available]
    cat_exist = [c for c in categorical_features if c in X.columns]
    num_exist = [c for c in numeric_features if c in X.columns]

    # Preparar target
    y_time = df['tempo_execucao']
    groups = df["dataset_id"]

    return X, y_time, groups, cat_exist, num_exist

def build_regression_pipelines(cat_cols: List[str], num_cols: List[str]) -> Dict[str, Pipeline]:
    """Constrói pipelines para os regressores."""
    
    # Pré-processamento comum
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", PowerTransformer(method='yeo-johnson'), num_cols)
        ],
        remainder="passthrough",
    )

    # Configurações dos modelos de regressão com log transform
    regressors = {
        "lightgbm": LGBMRegressor(
            objective="regression",
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbose=-1
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=2000,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
        )
    }

    # Criar pipelines de regressão com log transform
    regressor_pipelines = {}
    for name, model in regressors.items():
        regressor_pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

    return regressor_pipelines

def train_validate_save():
    """Treina e valida os modelos de regressão, salvando apenas o melhor regressor."""
    X, y_time, groups, cat_cols, num_cols = load_dataset(CSV_PATH)
    regressor_pipelines = build_regression_pipelines(cat_cols, num_cols)
    
    # Criar diretório para modelos se não existir
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)

    # Métricas para cada modelo
    regressor_scores = {}
    best_r2 = -float('inf')
    champion_regressor = None
    champion_regressor_name = None
    
    print("\n===== TREINAMENTO E VALIDAÇÃO DOS MODELOS DE REGRESSÃO =====")
    # Log transform do target para garantir previsões positivas
    y_time_log = np.log1p(y_time)
    
    for name, pipeline in regressor_pipelines.items():
        print(f"\n----- {name.upper()} -----")
        
        cv = GroupKFold(n_splits=5)
        scores = cross_validate(
            pipeline,
            X,
            y_time_log,  # Usar target log transformado
            groups=groups,
            cv=cv,
            scoring=(
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "r2",
            ),
            n_jobs=1,
            verbose=1,
            return_train_score=False,
        )

        mae_folds = -scores["test_neg_mean_absolute_error"]
        rmse_folds = -scores["test_neg_root_mean_squared_error"]
        r2_folds = scores["test_r2"]

        print(f"\nMétricas por fold:")
        for i in range(len(mae_folds)):
            print(f"Fold {i+1}:  MAE={mae_folds[i]:,.2f}  "
                  f"RMSE={rmse_folds[i]:,.2f}  R²={r2_folds[i]:.2f}")

        print(f"\nMétricas médias:")
        print(f"MAE  : {mae_folds.mean():,.2f} ± {mae_folds.std():.2f}")
        print(f"RMSE : {rmse_folds.mean():,.2f} ± {rmse_folds.std():.2f}")
        print(f"R²   : {r2_folds.mean():.2f} ± {r2_folds.std():.2f}")

        # Salvar métricas
        regressor_scores[name] = {
            'mae_mean': mae_folds.mean(),
            'mae_std': mae_folds.std(),
            'rmse_mean': rmse_folds.mean(),
            'rmse_std': rmse_folds.std(),
            'r2_mean': r2_folds.mean(),
            'r2_std': r2_folds.std()
        }

        # Verificar se é o melhor modelo até agora
        if r2_folds.mean() > best_r2:
            best_r2 = r2_folds.mean()
            champion_regressor = pipeline
            champion_regressor_name = name

    # Treinar e salvar os modelos
    if champion_regressor is not None:
        print(f"\n===== TREINANDO MODELO FINAL =====")
        print(f"Regressor Campeão: {champion_regressor_name.upper()}")
        
        # Treinar regressor com target log transformado
        champion_regressor.fit(X, y_time_log)
        
        # Salvar modelo
        joblib.dump(champion_regressor, os.path.join(MODELS_OUT_DIR, "modelo_regressor.pkl"), compress=3)
        
        # Salvar métricas
        metrics = {
            'regressor': {
                'name': champion_regressor_name,
                'metrics': regressor_scores[champion_regressor_name]
            }
        }
        joblib.dump(metrics, os.path.join(MODELS_OUT_DIR, "metricas_modelos.pkl"), compress=3)
        
        # Salvar features
        joblib.dump(X.columns.tolist(), FEATURES_OUT, compress=3)
        
        print("\n===== RESUMO FINAL =====")
        print(f"Regressor Campeão: {champion_regressor_name.upper()}")
        print(f"MAE  : {regressor_scores[champion_regressor_name]['mae_mean']:,.2f} ± "
              f"{regressor_scores[champion_regressor_name]['mae_std']:,.2f}")
        print(f"RMSE : {regressor_scores[champion_regressor_name]['rmse_mean']:,.2f} ± "
              f"{regressor_scores[champion_regressor_name]['rmse_std']:,.2f}")
        print(f"R²   : {regressor_scores[champion_regressor_name]['r2_mean']:.2f} ± "
              f"{regressor_scores[champion_regressor_name]['r2_std']:.2f}")

if __name__ == "__main__":
    train_validate_save()
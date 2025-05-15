"""
Treina um modelo de Random Forest para predição de falhas baseado em uso de memória.
Usa Random Forest com ajustes para melhor generalização e interpretabilidade.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import os

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
CSV_PATH = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/benchmark_WIN-KIOBB81FP3L_20250514_114302.csv"
)
MODELS_OUT_DIR = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
)
MEMORY_FEATURES_OUT = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/"
    "forest_memory_features.pkl"
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa os dados focando em features de memória."""
    df = df.copy()
    
    # Feature principal de proporção de memória
    df['proporcao_memoria'] = df['tamanho_dataset_nominal_mb'] / df['memoria_total_mb']
    
    return df

def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Carrega e prepara o dataset para treinamento."""
    df = pd.read_csv(path)

    if "dataset_id" not in df.columns:
        raise KeyError("Coluna 'dataset_id' não encontrada no CSV — necessária para GroupKFold.")
    
    # Verificar e limpar dados
    print("\nVerificando dados de entrada:")
    print(f"Total de registros: {len(df)}")
    print(f"Registros com status 'completed': {df['status'].eq('completed').sum()}")
    print(f"Registros com status 'failed': {df['status'].eq('failed').sum()}")
    
    # Remover registros com valores nulos em features importantes
    important_cols = ['status', 'biblioteca', 'dataset_formato', 
                     'tamanho_dataset_nominal_mb', 'memoria_total_mb']
    df = df.dropna(subset=important_cols)
    print(f"Registros após remoção de nulos: {len(df)}")
    
    # Pré-processar dados
    df = preprocess_data(df)
    
    # Preparar features
    categorical_features = [
        "biblioteca", 
        "dataset_formato"
    ]
    
    memory_features = [
        "tamanho_dataset_nominal_mb",
        "memoria_total_mb",
        "proporcao_memoria"
    ]

    # Verificar features disponíveis
    available = [c for c in categorical_features + memory_features if c in df.columns]
    print(f"\nFeatures disponíveis: {len(available)}")
    print(f"Features categóricas: {[c for c in categorical_features if c in available]}")
    print(f"Features de memória: {[c for c in memory_features if c in available]}")
    
    X = df[available]
    cat_exist = [c for c in categorical_features if c in X.columns]
    memory_exist = [c for c in memory_features if c in X.columns]

    # Preparar target
    y_class = (df['status'] == 'completed').astype(int)  # 1 = sucesso, 0 = falha
    groups = df["dataset_id"]
    
    # Verificar balanceamento das classes
    print(f"\nBalanceamento das classes:")
    print(f"Sucesso (1): {y_class.sum()} ({y_class.mean():.1%})")
    print(f"Falha (0): {len(y_class) - y_class.sum()} ({1 - y_class.mean():.1%})")

    return X, y_class, groups, cat_exist, memory_exist

def build_forest_classifier(cat_cols: List[str], memory_cols: List[str]) -> Pipeline:
    """Constrói o pipeline para o classificador Random Forest."""
    
    # Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("memory", StandardScaler(), memory_cols)
        ],
        remainder="passthrough",
    )

    # Classificador Random Forest
    forest_classifier = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight={0: 2.0, 1: 1.0},  # Peso maior para falhas
            random_state=42,
            n_jobs=1
        ))
    ])

    return forest_classifier

def train_validate_save():
    """Treina, valida e salva o modelo de classificação Random Forest."""
    X, y_class, groups, cat_cols, memory_cols = load_dataset(CSV_PATH)
    forest_classifier = build_forest_classifier(cat_cols, memory_cols)
    
    # Criar diretório para modelos se não existir
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    
    print("\n===== VALIDAÇÃO CRUZADA DO CLASSIFICADOR RANDOM FOREST =====")
    
    # Validação cruzada
    cv = GroupKFold(n_splits=5)
    scores = cross_validate(
        forest_classifier,
        X,
        y_class,
        groups=groups,
        cv=cv,
        scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )
    
    # Exibir métricas por fold
    print("\nMétricas por fold:")
    for i in range(len(scores["test_accuracy"])):
        print(f"Fold {i+1}: "
              f"Accuracy={scores['test_accuracy'][i]:.3f}, "
              f"Precision={scores['test_precision'][i]:.3f}, "
              f"Recall={scores['test_recall'][i]:.3f}, "
              f"F1={scores['test_f1'][i]:.3f}, "
              f"AUC={scores['test_roc_auc'][i]:.3f}")
    
    # Exibir métricas médias
    print("\nMétricas médias da validação cruzada:")
    print(f"Accuracy: {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"Precision: {scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}")
    print(f"Recall: {scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}")
    print(f"F1: {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"AUC: {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")
    
    print("\n===== TREINAMENTO DO MODELO FINAL =====")
    # Treinar o modelo final
    forest_classifier.fit(X, y_class)
    
    # Avaliar no conjunto de treino
    y_pred = forest_classifier.predict(X)
    y_prob = forest_classifier.predict_proba(X)[:, 1]
    
    # Threshold conservador
    memory_threshold = 0.4
    y_pred_conservative = (y_prob >= memory_threshold).astype(int)
    
    print("\nMétricas do classificador no conjunto de treino (threshold conservador 0.4):")
    print(f"Accuracy: {accuracy_score(y_class, y_pred_conservative):.3f}")
    print(f"Precision: {precision_score(y_class, y_pred_conservative):.3f}")
    print(f"Recall: {recall_score(y_class, y_pred_conservative):.3f}")
    print(f"F1: {f1_score(y_class, y_pred_conservative):.3f}")
    print(f"AUC: {roc_auc_score(y_class, y_prob):.3f}")
    
    # Salvar modelo e features
    joblib.dump(forest_classifier, os.path.join(MODELS_OUT_DIR, "forest_memory_classifier.pkl"), compress=3)
    joblib.dump(memory_threshold, os.path.join(MODELS_OUT_DIR, "forest_memory_threshold.pkl"), compress=3)
    joblib.dump(X.columns.tolist(), MEMORY_FEATURES_OUT, compress=3)
    
    # Salvar métricas
    metrics = {
        'forest_memory_classifier': {
            'name': 'random_forest',
            'threshold': memory_threshold,
            'metrics': {
                'accuracy': accuracy_score(y_class, y_pred_conservative),
                'precision': precision_score(y_class, y_pred_conservative),
                'recall': recall_score(y_class, y_pred_conservative),
                'f1': f1_score(y_class, y_pred_conservative),
                'auc': roc_auc_score(y_class, y_prob)
            }
        }
    }
    joblib.dump(metrics, os.path.join(MODELS_OUT_DIR, "forest_memory_metrics.pkl"), compress=3)
    
    # Análise de importância das features
    print("\n===== ANÁLISE DE IMPORTÂNCIA DAS FEATURES =====")
    feature_names = []
    
    # Nomes das features categóricas após one-hot encoding
    if cat_cols:
        ohe = forest_classifier.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names.extend(cat_feature_names)
    
    # Adicionar nomes das features de memória
    feature_names.extend(memory_cols)
    
    # Obter importância das features
    importances = forest_classifier.named_steps['classifier'].feature_importances_
    
    # Ordenar por importância
    importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 features mais importantes:")
    for name, importance in importance[:10]:
        print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    train_validate_save() 
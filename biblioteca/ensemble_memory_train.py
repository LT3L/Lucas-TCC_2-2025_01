"""
Treina modelos de classificação para predição de falhas baseado em uso de memória.
Compara modelos Linear (Logistic Regression) e Random Forest, selecionando o melhor
como modelo campeão baseado nas métricas de validação cruzada.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import os
import glob

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
CSV_PATH = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/benchmarks/benchmark_*.csv"
)
MODELS_OUT_DIR = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
)
MEMORY_FEATURES_OUT = (
    "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"
    "ensemble_memory_features.pkl"
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa os dados focando em features de memória."""
    df = df.copy()
    
    # Feature principal de proporção de memória
    df['proporcao_memoria'] = df['tamanho_dataset_nominal_mb'] / df['memoria_total_mb']
    
    return df

def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Carrega e prepara o dataset para treinamento."""
    # Ler todos os arquivos CSV da pasta
    all_files = glob.glob(path)
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # Combinar todos os dataframes
    df = pd.concat(dfs, ignore_index=True)

    if "dataset_id" not in df.columns:
        raise KeyError("Coluna 'dataset_id' não encontrada no CSV — necessária para GroupKFold.")
    
    # Verificar e limpar dados
    print("\nVerificando dados de entrada:")
    print(f"Total de arquivos processados: {len(all_files)}")
    print(f"Total de registros: {len(df)}")
    print(f"Registros com status 'completed': {df['status'].eq('completed').sum()}")
    print(f"Registros com status 'failed': {df['status'].eq('failed').sum()}")
    
    # Remover registros com erro para datasets menores que 10000MB
    df = df[~((df['status'] != 'completed') & (df['tamanho_dataset_nominal_mb'] < 10000))]
    print(f"\nRegistros após remoção de erros em datasets < 10000MB: {len(df)}")
    
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

def build_linear_classifier(cat_cols: List[str], memory_cols: List[str]) -> Pipeline:
    """Constrói o pipeline para o classificador linear."""
    
    # Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("memory", StandardScaler(), memory_cols)
        ],
        remainder="passthrough",
    )

    # Classificador Linear com regularização
    linear_classifier = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,  # Força da regularização (inverso)
            class_weight={0: 2.0, 1: 1.0},  # Peso maior para falhas
            max_iter=1000,
            random_state=42,
            n_jobs=1
        ))
    ])

    return linear_classifier

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

def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                  model_name: str) -> Dict:
    """Avalia um modelo usando validação cruzada e retorna as métricas."""
    print(f"\n===== VALIDAÇÃO CRUZADA DO {model_name.upper()} =====")
    
    # Validação cruzada
    cv = GroupKFold(n_splits=5)
    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )
    
    # Exibir métricas por fold
    print(f"\nMétricas por fold ({model_name}):")
    for i in range(len(scores["test_accuracy"])):
        print(f"Fold {i+1}: "
              f"Accuracy={scores['test_accuracy'][i]:.3f}, "
              f"Precision={scores['test_precision'][i]:.3f}, "
              f"Recall={scores['test_recall'][i]:.3f}, "
              f"F1={scores['test_f1'][i]:.3f}, "
              f"AUC={scores['test_roc_auc'][i]:.3f}")
    
    # Calcular métricas médias
    metrics = {
        'accuracy': scores['test_accuracy'].mean(),
        'accuracy_std': scores['test_accuracy'].std(),
        'precision': scores['test_precision'].mean(),
        'precision_std': scores['test_precision'].std(),
        'recall': scores['test_recall'].mean(),
        'recall_std': scores['test_recall'].std(),
        'f1': scores['test_f1'].mean(),
        'f1_std': scores['test_f1'].std(),
        'auc': scores['test_roc_auc'].mean(),
        'auc_std': scores['test_roc_auc'].std()
    }
    
    # Exibir métricas médias
    print(f"\nMétricas médias da validação cruzada ({model_name}):")
    print(f"Accuracy: {metrics['accuracy']:.3f} ± {metrics['accuracy_std']:.3f}")
    print(f"Precision: {metrics['precision']:.3f} ± {metrics['precision_std']:.3f}")
    print(f"Recall: {metrics['recall']:.3f} ± {metrics['recall_std']:.3f}")
    print(f"F1: {metrics['f1']:.3f} ± {metrics['f1_std']:.3f}")
    print(f"AUC: {metrics['auc']:.3f} ± {metrics['auc_std']:.3f}")
    
    return metrics

def select_champion_model(linear_metrics: Dict, forest_metrics: Dict) -> str:
    """Seleciona o melhor modelo baseado nas métricas de validação cruzada."""
    print("\n===== SELEÇÃO DO MODELO CAMPEÃO =====")
    
    # Critério principal: F1-Score (balanceia precision e recall)
    # Critério secundário: AUC (área sob a curva ROC)
    # Critério terciário: Accuracy
    
    linear_f1 = linear_metrics['f1']
    forest_f1 = forest_metrics['f1']
    
    linear_auc = linear_metrics['auc']
    forest_auc = forest_metrics['auc']
    
    linear_acc = linear_metrics['accuracy']
    forest_acc = forest_metrics['accuracy']
    
    print(f"Linear F1-Score: {linear_f1:.3f}")
    print(f"Forest F1-Score: {forest_f1:.3f}")
    print(f"Linear AUC: {linear_auc:.3f}")
    print(f"Forest AUC: {forest_auc:.3f}")
    print(f"Linear Accuracy: {linear_acc:.3f}")
    print(f"Forest Accuracy: {forest_acc:.3f}")
    
    # Comparar F1-Score
    if abs(linear_f1 - forest_f1) < 0.01:  # Diferença menor que 1%
        # Se F1 muito próximo, usar AUC como critério
        if abs(linear_auc - forest_auc) < 0.01:  # Diferença menor que 1%
            # Se AUC também muito próximo, usar accuracy
            if linear_acc >= forest_acc:
                champion = "linear"
                print(f"\n🏆 MODELO CAMPEÃO: Linear (critério: accuracy)")
            else:
                champion = "forest"
                print(f"\n🏆 MODELO CAMPEÃO: Random Forest (critério: accuracy)")
        else:
            if linear_auc > forest_auc:
                champion = "linear"
                print(f"\n🏆 MODELO CAMPEÃO: Linear (critério: AUC)")
            else:
                champion = "forest"
                print(f"\n🏆 MODELO CAMPEÃO: Random Forest (critério: AUC)")
    else:
        if linear_f1 > forest_f1:
            champion = "linear"
            print(f"\n🏆 MODELO CAMPEÃO: Linear (critério: F1-Score)")
        else:
            champion = "forest"
            print(f"\n🏆 MODELO CAMPEÃO: Random Forest (critério: F1-Score)")
    
    return champion

def analyze_model_features(model: Pipeline, cat_cols: List[str], memory_cols: List[str], 
                          model_name: str):
    """Analisa e exibe as features mais importantes do modelo."""
    print(f"\n===== ANÁLISE DE FEATURES ({model_name.upper()}) =====")
    
    feature_names = []
    
    # Nomes das features categóricas após one-hot encoding
    if cat_cols:
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names.extend(cat_feature_names)
    
    # Adicionar nomes das features de memória
    feature_names.extend(memory_cols)
    
    if model_name == "linear":
        # Para modelo linear, usar coeficientes
        coefficients = model.named_steps['classifier'].coef_[0]
        importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 10 features mais importantes (coeficientes):")
        for name, coef in importance[:10]:
            print(f"{name}: {coef:.4f}")
    else:
        # Para Random Forest, usar feature_importances_
        importances = model.named_steps['classifier'].feature_importances_
        importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 features mais importantes:")
        for name, importance in importance[:10]:
            print(f"{name}: {importance:.4f}")

def train_validate_save():
    """Treina, valida e salva os modelos, selecionando o melhor como campeão."""
    X, y_class, groups, cat_cols, memory_cols = load_dataset(CSV_PATH)
    
    # Criar diretório para modelos se não existir
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    
    # Construir modelos
    linear_classifier = build_linear_classifier(cat_cols, memory_cols)
    forest_classifier = build_forest_classifier(cat_cols, memory_cols)
    
    # Avaliar modelos
    linear_metrics = evaluate_model(linear_classifier, X, y_class, groups, "Linear")
    forest_metrics = evaluate_model(forest_classifier, X, y_class, groups, "Random Forest")
    
    # Selecionar modelo campeão
    champion = select_champion_model(linear_metrics, forest_metrics)
    
    # Treinar modelo final do campeão
    print(f"\n===== TREINAMENTO DO MODELO CAMPEÃO ({champion.upper()}) =====")
    
    if champion == "linear":
        final_model = linear_classifier
        final_metrics = linear_metrics
    else:
        final_model = forest_classifier
        final_metrics = forest_metrics
    
    # Nomes genéricos para o modelo campeão
    model_filename = "champion_success_predictor.pkl"
    threshold_filename = "champion_success_threshold.pkl"
    metrics_filename = "champion_success_metrics.pkl"
    
    # Treinar o modelo final
    final_model.fit(X, y_class)
    
    # Avaliar no conjunto de treino
    y_pred = final_model.predict(X)
    y_prob = final_model.predict_proba(X)[:, 1]
    
    # Threshold conservador
    memory_threshold = 0.4
    y_pred_conservative = (y_prob >= memory_threshold).astype(int)
    
    print(f"\nMétricas do modelo campeão no conjunto de treino (threshold conservador 0.4):")
    print(f"Accuracy: {accuracy_score(y_class, y_pred_conservative):.3f}")
    print(f"Precision: {precision_score(y_class, y_pred_conservative):.3f}")
    print(f"Recall: {recall_score(y_class, y_pred_conservative):.3f}")
    print(f"F1: {f1_score(y_class, y_pred_conservative):.3f}")
    print(f"AUC: {roc_auc_score(y_class, y_prob):.3f}")
    
    # Salvar apenas os arquivos essenciais para uso do modelo campeão
    print(f"\n===== SALVANDO MODELO CAMPEÃO =====")
    
    # 1. Modelo treinado (essencial)
    joblib.dump(final_model, os.path.join(MODELS_OUT_DIR, model_filename), compress=3)
    print(f"✅ Modelo salvo: {model_filename}")
    
    # 2. Threshold para predições (essencial)
    joblib.dump(memory_threshold, os.path.join(MODELS_OUT_DIR, threshold_filename), compress=3)
    print(f"✅ Threshold salvo: {threshold_filename}")
    
    # 3. Lista de features esperadas (essencial)
    joblib.dump(X.columns.tolist(), MEMORY_FEATURES_OUT, compress=3)
    print(f"✅ Features salvas: ensemble_memory_features.pkl")
    
    # Analisar features do modelo campeão
    analyze_model_features(final_model, cat_cols, memory_cols, champion)
    
    # Resumo final
    print(f"\n===== RESUMO FINAL =====")
    print(f"🏆 CAMPEÃO DE ESTIMATIVA DE SUCESSO TREINADO")
    print(f"Modelo base selecionado: {champion.upper()}")
    print(f"F1-Score (CV): {final_metrics['f1']:.3f} ± {final_metrics['f1_std']:.3f}")
    print(f"AUC (CV): {final_metrics['auc']:.3f} ± {final_metrics['auc_std']:.3f}")
    print(f"Accuracy (CV): {final_metrics['accuracy']:.3f} ± {final_metrics['accuracy_std']:.3f}")
    print(f"\n📁 ARQUIVOS ESSENCIAIS PARA USO DO MODELO:")
    print(f"   • {model_filename} (modelo treinado)")
    print(f"   • {threshold_filename} (threshold para predições)")
    print(f"   • ensemble_memory_features.pkl (lista de features)")
    print(f"\n💡 Use estes 3 arquivos para fazer predições de sucesso!")

if __name__ == "__main__":
    train_validate_save() 
"""
Treina e valida um modelo LightGBM com GroupKFold (k=5).
Salva o pipeline completo via Joblib.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor


# --------------------------------------------------------------------------- #
# 1. CAMINHOS ESTÁTICOS                                                       #
# --------------------------------------------------------------------------- #
CSV_PATH = (
    "/Users/lucas.lima/Documents/Projects/TCC_2/app/datasets_and_models_output /benchmark_Factored-K7X5WF003P_20250428_210801.csv"
)
MODEL_OUT = (
    "/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/"
    "modelo_lightgbm.pkl"
)
FEATURES_OUT = (
    "/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/"
    "modelo_lightgbm_features.pkl"
)


# --------------------------------------------------------------------------- #
# 2. CARREGAMENTO E FEATURES                                                  #
# --------------------------------------------------------------------------- #
def load_dataset(path: str):
    df = pd.read_csv(path)

    # **IMPORTANTE**: cada linha deve ter um identificador
    # (mesmo dataset em tamanhos/formatos diferentes → mesmo id)
    if "dataset_id" not in df.columns:
        raise KeyError("Coluna 'dataset_id' não encontrada no CSV — "
                       "necessária para GroupKFold.")

    y = df["tempo_execucao"]

    categorical_features = ["biblioteca", "dataset_formato"]
    numeric_features = [
        "tamanho_dataset_nominal_mb",
        "tamanho_dataset_bytes",
        "num_linhas",
        "num_colunas",
        "percentual_numerico",
        "percentual_string",
        "percentual_datetime",
        "nucleos_fisicos",
        "nucleos_logicos",
        "frequencia_cpu_max",
        "memoria_total_mb",
        "disco_total_gb",
        "tem_joins",
        "tem_groupby",
    ]

    available = [c for c in categorical_features + numeric_features if c in df.columns]
    X = df[available]

    cat_exist = [c for c in categorical_features if c in X.columns]
    num_exist = [c for c in numeric_features if c in X.columns]

    groups = df["dataset_id"]

    return X, y, groups, cat_exist, num_exist


# --------------------------------------------------------------------------- #
# 3. PIPELINE (PRÉ + LIGHTGBM)                                                #
# --------------------------------------------------------------------------- #
def build_pipeline(cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )

    lgbm = LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("regressor", lgbm)])


# --------------------------------------------------------------------------- #
# 4. VALIDAR COM GROUPKFOLD, TREINAR FINAL E SALVAR                           #
# --------------------------------------------------------------------------- #
def train_validate_save():
    X, y, groups, cat_cols, _ = load_dataset(CSV_PATH)
    pipeline = build_pipeline(cat_cols)

    cv = GroupKFold(n_splits=5)
    scores = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=(
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
            "r2",
        ),
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    print("\n===== MÉTRICAS (GroupKFold 5) =====")
    mae_folds = -scores["test_neg_mean_absolute_error"]
    rmse_folds = -scores["test_neg_root_mean_squared_error"]
    r2_folds = scores["test_r2"]

    for i in range(len(mae_folds)):
        print(f"Fold {i+1}:  MAE={mae_folds[i]:,.2f}  "
              f"RMSE={rmse_folds[i]:,.2f}  R²={r2_folds[i]:.2f}")

    print(f"\nMAE médio  : {mae_folds.mean():,.2f} ± {mae_folds.std():.2f}")
    print(f"RMSE médio : {rmse_folds.mean():,.2f} ± {rmse_folds.std():.2f}")
    print(f"R²  médio  : {r2_folds.mean():.2f} ± {r2_folds.std():.2f}")

    # Treina em TODO o conjunto para exportar o modelo final
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_OUT)
    joblib.dump(X.columns.tolist(), FEATURES_OUT)

    print(f"\nModelo final salvo em: {MODEL_OUT}")
    print(f"Lista de features    : {FEATURES_OUT}")


# --------------------------------------------------------------------------- #
# 5. EXECUÇÃO                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train_validate_save()
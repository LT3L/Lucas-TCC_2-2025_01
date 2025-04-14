import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar o CSV com os dados reais
df = pd.read_csv("/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/dataset_benchmark.csv")

# Definir a variável alvo
y = df["tempo_execucao"]

# Definir features categóricas e numéricas
categorical_features = ["biblioteca", "dataset_formato"]
numeric_features = [
    "tamanho_dataset_nominal_mb", "tamanho_dataset_bytes", "num_linhas", "num_colunas",
    "percentual_numerico", "percentual_string", "percentual_datetime",
    "nucleos_fisicos", "nucleos_logicos", "frequencia_cpu_max",
    "memoria_total_mb", "disco_total_gb",
    "tem_joins", "tem_groupby"
]

# Garantir que as colunas existem no CSV
features_existentes = [col for col in categorical_features + numeric_features if col in df.columns]
X = df[features_existentes]

# Criar pré-processador para as variáveis categóricas
categorical_cols_existentes = [col for col in categorical_features if col in X.columns]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols_existentes)
    ],
    remainder="passthrough"
)

# Criar pipeline de modelo com pré-processamento + regressão
modelo = Pipeline(steps=[
    ("preprocessador", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo.fit(X_train, y_train)

# Avaliar o modelo
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Resultados no conjunto de teste:")
print(f"  - MAE: {mae:.2f}")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - R²: {r2:.2f}")

# Salvar o modelo e a ordem das features
joblib.dump(modelo, "/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/modelo_regressao_tempo_execucao.pkl")
joblib.dump(X.columns.tolist(), "/Users/lucas.lima/Documents/Projects/TCC_2/datasets_and_models_output/modelo_regressao_features.pkl")
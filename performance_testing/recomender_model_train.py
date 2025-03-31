import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Reutilizar o DataFrame gerado anteriormente

df_raw = pd.read_csv("../datasets_and_models_output/dataset_benchmark.csv")

df = df_raw.copy()

# A variável alvo será a biblioteca com melhor desempenho (menor tempo_execucao)
# Para fins de treino, assumimos que a melhor biblioteca é aquela com menor tempo por dataset/configuração
# Primeiro vamos ordenar e marcar a melhor opção

df["chave_config"] = (
    df["dataset_nome"] + "_" +
    df["dataset_formato"] + "_" +
    df["tamanho_dataset_nominal_mb"].astype(str)
)

# Obter a melhor biblioteca por configuração (menor tempo_execucao)
idx_melhor = df.groupby("chave_config")["tempo_execucao"].idxmin()
df["melhor_opcao"] = False
df.loc[idx_melhor, "melhor_opcao"] = True

# Nosso target será 'biblioteca' onde 'melhor_opcao' é True
df_target = df[df["melhor_opcao"]].copy()
y = df_target["biblioteca"]

# Selecionar apenas colunas numéricas como features
feature_cols = [
    "tamanho_dataset_nominal_mb", "cpu_medio_execucao", "memoria_media_execucao",
    "leitura_bytes", "escrita_bytes", "nucleos_fisicos", "nucleos_logicos",
    "frequencia_cpu_max", "memoria_total_mb", "disco_total_gb",
    "tamanho_dataset_bytes", "num_linhas", "num_colunas",
    "percentual_numerico", "percentual_string", "percentual_datetime"
]
X = df_target[feature_cols]

# Codificar a variável alvo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Salvar o modelo e o codificador
joblib.dump(feature_cols, "../datasets_and_models_output/modelo_features.pkl")
joblib.dump(modelo, "../datasets_and_models_output/modelo_recomendador.pkl")
joblib.dump(label_encoder, "../datasets_and_models_output/label_encoder.pkl")

print("Model output successfully created")
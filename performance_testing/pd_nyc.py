import pandas as pd
import numpy as np
import argparse
import os

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script Pandas NYC")
parser.add_argument('--input', type=str, required=True, help='Caminho para o arquivo de entrada')
args = parser.parse_args()

file = args.input

# Verificação extra para debug
# print(f"📄 Lendo arquivo: {file}")
# print(f"📁 Existe? {os.path.exists(file)}")

if file.endswith(".parquet"):
    df = pd.read_parquet(file)

elif file.endswith(".json"):
    df = pd.read_json(file, lines=True)

else:
    df = pd.read_csv(file)

# Remover valores nulos
df.dropna(inplace=True)

# Ajustar os nomes das colunas de datetime
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

# Criar uma nova coluna com a duração da viagem (em minutos)
df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

# Filtrar viagens inválidas (tempo negativo, passageiros <= 0, distância <= 0)
df = df[(df["trip_duration"] > 0) & (df["passenger_count"] > 0) & (df["trip_distance"] > 0)]

# Criar colunas adicionais
df["average_speed_kmh"] = df["trip_distance"] / (df["trip_duration"] / 60)
df["fare_per_km"] = df["fare_amount"] / df["trip_distance"]

# Criar colunas de data para agregação
df["year"] = df["tpep_pickup_datetime"].dt.year
df["month"] = df["tpep_pickup_datetime"].dt.month
df["day"] = df["tpep_pickup_datetime"].dt.day
df["hour"] = df["tpep_pickup_datetime"].dt.hour

# Estatísticas básicas
descriptive_stats = df[["trip_distance", "fare_amount", "trip_duration", "average_speed_kmh", "fare_per_km"]].describe()

# Salvar resultados
output_folder = "processed_data/"

descriptive_stats.to_csv(output_folder + "descriptive_stats.csv")

# Salvar os dados processados em formato Parquet
df.to_parquet(output_folder + "nyc_taxi_processed.parquet", index=False)

# Exibir amostra dos dados processados
# print(df.head())
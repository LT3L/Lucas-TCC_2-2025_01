import duckdb
import argparse
import os
from pathlib import Path
from glob import glob
import pandas as pd

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script DuckDB NYC")
parser.add_argument('--input', type=str, required=True, help='Caminho para o diretório de entrada')
args = parser.parse_args()

diretorio = args.input

def detectar_formato_arquivo(diretorio):
    """Detecta o formato do arquivo baseado na extensão do primeiro arquivo encontrado"""
    # Procura por arquivos em ordem: csv, json, parquet
    for ext in ['.csv', '.json', '.parquet']:
        arquivos = glob(os.path.join(diretorio, f'*{ext}'))
        if arquivos:
            return ext.replace('.', '')
    return None

def ler_arquivo(caminho, formato):
    """Lê o arquivo no formato especificado"""
    if formato == 'csv':
        df = duckdb.sql(f"SELECT * FROM read_csv_auto('{caminho}/*.csv')").df()
    elif formato == 'json':
        df = duckdb.sql(f"SELECT * FROM read_json_auto('{caminho}/*.json')").df()
    elif formato == 'parquet':
        df = duckdb.sql(f"SELECT * FROM '{caminho}/*.parquet'").df()
    else:
        raise ValueError(f"Formato não suportado: {formato}")
    
    return df

# Detecta o formato e lê os arquivos
formato = detectar_formato_arquivo(diretorio)
if not formato:
    print(f"Nenhum arquivo válido encontrado em {diretorio}")
    exit(1)

print(f"Formato detectado: {formato}")
df = ler_arquivo(diretorio, formato)

# Criar conexão para processamento SQL
con = duckdb.connect()
con.register("trips", df)

# Processamento SQL (filtros e colunas derivadas)

if formato == 'csv' or formato == 'parquet':
    query = """
    SELECT *,
        CAST(tpep_dropoff_datetime AS TIMESTAMP) - CAST(tpep_pickup_datetime AS TIMESTAMP) AS trip_duration_interval,
        EXTRACT(YEAR FROM CAST(tpep_pickup_datetime AS TIMESTAMP)) AS year,
        EXTRACT(MONTH FROM CAST(tpep_pickup_datetime AS TIMESTAMP)) AS month,
        EXTRACT(DAY FROM CAST(tpep_pickup_datetime AS TIMESTAMP)) AS day,
        EXTRACT(HOUR FROM CAST(tpep_pickup_datetime AS TIMESTAMP)) AS hour
    FROM trips
    WHERE passenger_count > 0 AND trip_distance > 0 AND
        CAST(tpep_dropoff_datetime AS TIMESTAMP) > CAST(tpep_pickup_datetime AS TIMESTAMP)
    """

if formato == 'json':
    query = """
    SELECT *,
        to_timestamp(tpep_dropoff_datetime) - to_timestamp(tpep_pickup_datetime) AS trip_duration_interval,
        EXTRACT(YEAR FROM to_timestamp(tpep_pickup_datetime)) AS year,
        EXTRACT(MONTH FROM to_timestamp(tpep_pickup_datetime)) AS month,
        EXTRACT(DAY FROM to_timestamp(tpep_pickup_datetime)) AS day,
        EXTRACT(HOUR FROM to_timestamp(tpep_pickup_datetime)) AS hour
    FROM trips
    WHERE passenger_count > 0 AND trip_distance > 0 AND
        tpep_dropoff_datetime - tpep_pickup_datetime > 0"""

df = con.execute(query).df()

# Cálculos adicionais no DataFrame
df["trip_duration"] = df["trip_duration_interval"].dt.total_seconds() / 60
df["average_speed_kmh"] = df["trip_distance"] / (df["trip_duration"] / 60)
df["fare_per_km"] = df["fare_amount"] / df["trip_distance"]

# Salvar arquivos de saída
output_folder = "processed_data/"
os.makedirs(output_folder, exist_ok=True)

# descriptive_stats.to_csv(os.path.join(output_folder, "descriptive_stats.csv"), index=False)
con.register("df", df)
con.execute(f"COPY df TO '{os.path.join(output_folder, 'nyc_taxi_processed.parquet')}' (FORMAT PARQUET)")

print(f"Arquivo salvo em {os.path.join(output_folder, 'nyc_taxi_processed.parquet')}")

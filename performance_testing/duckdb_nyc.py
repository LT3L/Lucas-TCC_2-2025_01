import duckdb
import argparse
import os
from pathlib import Path

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script DuckDB NYC")
parser.add_argument('--input', type=str, required=True, help='Caminho para o arquivo de entrada')
args = parser.parse_args()

file = args.input
ext = Path(file).suffix.lower().replace('.', '')

# Leitura do arquivo com DuckDB
if ext == "parquet":
    df = duckdb.sql(f"SELECT * FROM '{file}'").df()
elif ext == "json":
    df = duckdb.sql(f"SELECT * FROM read_json_auto('{file}')").df()
else:
    df = duckdb.sql(f"SELECT * FROM read_csv_auto('{file}')").df()

# Criar conexão para processamento SQL
con = duckdb.connect()
con.register("trips", df)

# Processamento SQL (filtros e colunas derivadas)
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
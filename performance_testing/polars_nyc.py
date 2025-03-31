import polars as pl
import argparse
import os

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script Polars NYC")
parser.add_argument('--input', type=str, required=True, help='Caminho para o(s) arquivo(s) de entrada')
args = parser.parse_args()

# Definir o caminho dos arquivos (pode ser um glob pattern)
file = args.input

if file.endswith(".parquet"):
    df = pl.read_parquet(file)
    df = df.with_columns(
        pl.col("tpep_pickup_datetime").cast(pl.Datetime("ms")),
        pl.col("tpep_dropoff_datetime").cast(pl.Datetime("ms"))
    )

elif file.endswith(".json") or file.endswith(".jsonl"):
    df = pl.read_ndjson(file)
    df = df.with_columns(
        (pl.col("tpep_pickup_datetime") * 1000).cast(pl.Datetime("ms")),
        (pl.col("tpep_dropoff_datetime") * 1000).cast(pl.Datetime("ms"))
    )

elif file.endswith(".csv"):
    df = pl.read_csv(file)
    df = df.with_columns(
        pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        pl.col("tpep_dropoff_datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )

else:
    print(f"❌ Formato de arquivo não suportado: {file}")
    raise "Error with file format"


# Remover nulos
df = df.drop_nulls()

# Criar coluna de duração da viagem (minutos)
df = df.with_columns(
    ((pl.col("tpep_dropoff_datetime").cast(pl.Int64) - pl.col("tpep_pickup_datetime").cast(pl.Int64)) / 60).alias("trip_duration")
)

# Adicionar colunas derivadas
df = df.with_columns(
    (pl.col("trip_distance") / (pl.col("trip_duration") / 60)).alias("average_speed_kmh"),
    (pl.col("fare_amount") / pl.col("trip_distance")).alias("fare_per_km"),
    pl.col("tpep_pickup_datetime").dt.year().alias("year"),
    pl.col("tpep_pickup_datetime").dt.month().alias("month"),
    pl.col("tpep_pickup_datetime").dt.day().alias("day"),
    pl.col("tpep_pickup_datetime").dt.hour().alias("hour")
)

# Filtrar viagens inválidas
df = df.filter(
    (pl.col("trip_duration") > 0) &
    (pl.col("passenger_count") > 0) &
    (pl.col("trip_distance") > 0)
)

# Criar pasta se não existir
output_folder = "processed_data/"
os.makedirs(output_folder, exist_ok=True)

# Salvar resultados
df.write_parquet(os.path.join(output_folder, "nyc_taxi_processed.parquet"))

# Exibir amostra
# print(df.head())
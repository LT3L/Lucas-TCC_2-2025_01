import polars as pl
import glob

# Definir o caminho dos arquivos Parquet
file_pattern = "yellow_tripdata_2024-*.parquet"
files = glob.glob(file_pattern)

# Carregar e padronizar datetime antes da concatenação
dfs = []
for file in files:
    df = pl.read_parquet(file)
    df = df.with_columns(
        pl.col("tpep_pickup_datetime").cast(pl.Datetime("ms")),
        pl.col("tpep_dropoff_datetime").cast(pl.Datetime("ms"))
    )
    dfs.append(df)

# Concatenar os arquivos após a padronização
df = pl.concat(dfs)

# Remover valores nulos
df = df.drop_nulls()

# Criar a coluna de duração da viagem separadamente
df = df.with_columns(
    ((pl.col("tpep_dropoff_datetime").cast(pl.Int64) - pl.col("tpep_pickup_datetime").cast(pl.Int64)) / 60).alias("trip_duration")
)

# Agora, adicionar as demais colunas
df = df.with_columns(
    (pl.col("trip_distance") / (pl.col("trip_duration") / 60)).alias("average_speed_kmh"),
    (pl.col("fare_amount") / pl.col("trip_distance")).alias("fare_per_km"),
    pl.col("tpep_pickup_datetime").dt.year().alias("year"),
    pl.col("tpep_pickup_datetime").dt.month().alias("month"),
    pl.col("tpep_pickup_datetime").dt.day().alias("day"),
    pl.col("tpep_pickup_datetime").dt.hour().alias("hour")
)

# Filtrar viagens inválidas
df = df.filter((pl.col("trip_duration") > 0) &
               (pl.col("passenger_count") > 0) &
               (pl.col("trip_distance") > 0))

# Estatísticas básicas
descriptive_stats = df.select(["trip_distance", "fare_amount", "trip_duration", "average_speed_kmh", "fare_per_km"]).describe()

# Salvar resultados
output_folder = "processed_data/"
descriptive_stats.write_csv(output_folder + "descriptive_stats.csv")
df.write_parquet(output_folder + "nyc_taxi_processed.parquet")

# Exibir amostra dos dados processados
print(df.head())
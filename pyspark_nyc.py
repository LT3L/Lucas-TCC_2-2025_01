from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, hour, unix_timestamp
import glob

# Criar sessão Spark
spark = SparkSession.builder.appName("NYC_Taxi_Processing").getOrCreate()

# Definir o caminho dos arquivos Parquet
file_pattern = "yellow_tripdata_2024-*.parquet"
files = glob.glob(file_pattern)

# Carregar os dados de múltiplos arquivos Parquet
df = spark.read.parquet(*files)

# Remover valores nulos
df = df.dropna()

# Criar colunas adicionais
df = df.withColumn("trip_duration",
                   (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60)
df = df.withColumn("average_speed_kmh", col("trip_distance") / (col("trip_duration") / 60))
df = df.withColumn("fare_per_km", col("fare_amount") / col("trip_distance"))

df = df.withColumn("year", year(col("tpep_pickup_datetime")))
df = df.withColumn("month", month(col("tpep_pickup_datetime")))
df = df.withColumn("day", dayofmonth(col("tpep_pickup_datetime")))
df = df.withColumn("hour", hour(col("tpep_pickup_datetime")))

# Filtrar viagens inválidas
df = df.filter((col("trip_duration") > 0) &
               (col("passenger_count") > 0) &
               (col("trip_distance") > 0))

# Estatísticas básicas
descriptive_stats = df.selectExpr(
    "percentile_approx(trip_distance, 0.5) as median_trip_distance",
    "percentile_approx(fare_amount, 0.5) as median_fare_amount",
    "percentile_approx(trip_duration, 0.5) as median_trip_duration",
    "percentile_approx(average_speed_kmh, 0.5) as median_speed",
    "percentile_approx(fare_per_km, 0.5) as median_fare_per_km"
)

# Salvar resultados
output_folder = "processed_data/"
descriptive_stats.write.csv(output_folder + "descriptive_stats.csv", header=True)
df.write.parquet(output_folder + "nyc_taxi_processed.parquet")

# Exibir amostra dos dados processados
df.show(5)

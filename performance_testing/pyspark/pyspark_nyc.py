from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, hour, unix_timestamp
import argparse
import os
import psutil
import gc

# Calculate available memory more conservatively
total_ram_gb = int(psutil.virtual_memory().total / (1024 ** 3))
# Use only 50% of available RAM for Spark
spark_memory_gb = max(1, int(total_ram_gb * 0.5))

spark = SparkSession.builder \
    .appName("Benchmark") \
    .master("local[*]") \
    .config("spark.driver.memory", f"{spark_memory_gb}g") \
    .config("spark.executor.memory", f"{spark_memory_gb}g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", f"{spark_memory_gb}g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .getOrCreate()

# Silenciar os WARNs
spark.sparkContext.setLogLevel("ERROR")

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script PySpark NYC")
parser.add_argument('--input', type=str, required=True, help='Caminho para o(s) arquivo(s) de entrada')
args = parser.parse_args()

# Definir o caminho dos arquivos (pode ser um glob pattern)
file = args.input

if file.endswith(".parquet"):
    df = spark.read.parquet(file)

elif file.endswith(".json") or file.endswith(".jsonl"):
    df = spark.read.json(file)

elif file.endswith(".csv"):
    df = spark.read.csv(file, header=True, inferSchema=True)

else:
    print(f"❌ Formato de arquivo não suportado: {file}")
    raise "Error with file format"


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

# Salvar resultados
output_folder = "processed_data/"
df.write.mode("overwrite").parquet(output_folder + "nyc_taxi_processed.parquet")

# Exibir amostra dos dados processados
# df.show(5)

# Add cleanup at the end
def cleanup():
    spark.stop()
    gc.collect()

# Register cleanup function
import atexit
atexit.register(cleanup)

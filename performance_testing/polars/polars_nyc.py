import polars as pl
import argparse
import os
import glob
from pathlib import Path

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script Polars NYC")
parser.add_argument('--input', type=str, required=True, help='Caminho para o diretório de entrada')
args = parser.parse_args()

# Lista todos os arquivos no diretório
arquivos = sorted(glob.glob(os.path.join(args.input, "*.*")))
if not arquivos:
    print(f"❌ Nenhum arquivo encontrado em {args.input}")
    exit(1)

# Verifica o formato do primeiro arquivo
primeiro_arquivo = arquivos[0]
extensao = Path(primeiro_arquivo).suffix.lower()

print(f"📁 Processando arquivos {extensao} do diretório: {args.input}")
print(f"📄 Arquivos encontrados: {len(arquivos)}")

# Lê todos os arquivos de uma vez
if extensao == '.parquet':
    # Parquet já tem os campos de datetime no formato correto
    df = pl.concat([pl.read_parquet(f) for f in arquivos])
elif extensao in ['.json', '.jsonl']:
    # Para JSON, os timestamps estão em formato Unix
    df = pl.concat([pl.read_ndjson(f) for f in arquivos])
    # Converte timestamps Unix para datetime
    df = df.with_columns([
        pl.from_epoch(pl.col("tpep_pickup_datetime").cast(pl.Int64)),
        pl.from_epoch(pl.col("tpep_dropoff_datetime").cast(pl.Int64))
    ])
elif extensao == '.csv':
    # Primeiro, lê um arquivo para inferir o schema
    schema = pl.read_csv(arquivos[0], try_parse_dates=True).schema
    # Lê todos os CSVs usando o mesmo schema
    df = pl.concat([pl.read_csv(f, schema=schema) for f in arquivos])
    # Não precisa converter datetime pois já está no formato correto
else:
    print(f"❌ Formato não suportado para o arquivo: {primeiro_arquivo}")
    exit(1)

print(f"📊 Total de registros lidos: {df.height}")

# Remove apenas linhas onde os campos essenciais são nulos
campos_essenciais = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance"]
df = df.drop_nulls(subset=campos_essenciais)
print(f"📊 Registros após remover nulos essenciais: {df.height}")

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
print(f"📊 Registros após filtrar viagens inválidas: {df.height}")

# Criar pasta se não existir
output_folder = "processed_data/"
os.makedirs(output_folder, exist_ok=True)

# Salvar resultados
output_file = os.path.join(output_folder, "nyc_taxi_processed.parquet")
df.write_parquet(output_file)
print(f"✅ Dados processados salvos em: {output_file}")

# Exibir estatísticas básicas
if df.height > 0:
    print("\n📈 Estatísticas básicas:")
    print(f"Total de viagens: {df.height}")
    print(f"Período: {df['year'].min()} a {df['year'].max()}")
    print(f"Distância média: {df['trip_distance'].mean():.2f} km")
    print(f"Valor médio: ${df['fare_amount'].mean():.2f}")
else:
    print("\n❌ Nenhum dado válido após processamento")
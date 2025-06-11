from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, year, avg, count, desc, to_date, to_timestamp, when, regexp_extract, trim, regexp_replace
import os
import argparse
import time

def processar_commits_github_spark(spark, diretorio_entrada):
    """Processa dados de commits do GitHub do diretório de entrada usando Spark"""
    
    # Lê todos os arquivos da pasta de uma vez
    if "parquet" in diretorio_entrada or any("parquet" in f for f in os.listdir(diretorio_entrada) if os.path.isfile(os.path.join(diretorio_entrada, f))):
        df = spark.read.parquet(f"{diretorio_entrada}/*.parquet")
        print(f"✅ Lidos arquivos Parquet da pasta: {df.count()} registros")

    elif "json" in diretorio_entrada or any("json" in f for f in os.listdir(diretorio_entrada) if os.path.isfile(os.path.join(diretorio_entrada, f))):
        df = spark.read.json(f"{diretorio_entrada}/*.json")
        print(f"✅ Lidos arquivos JSON da pasta: {df.count()} registros")

    elif "csv" in diretorio_entrada or any("csv" in f for f in os.listdir(diretorio_entrada) if os.path.isfile(os.path.join(diretorio_entrada, f))):
        df = spark.read.csv(f"{diretorio_entrada}/*.csv", header=True, inferSchema=True)
        print(f"✅ Lidos arquivos CSV da pasta: {df.count()} registros")

    else:
        print(f"❌ Formato de arquivo não suportado no diretório: {diretorio_entrada}")
        raise "Error with file format"
    
    # Converte timestamp
    if 'commit_time_utc' in df.columns:
        df = df.withColumn("commit_time_utc", 
                          when(col("commit_time_utc").rlike("^\\d+$"), 
                               to_timestamp(col("commit_time_utc").cast("long")))
                          .otherwise(to_timestamp(col("commit_time_utc"))))
    
    # Processa repository
    if 'repository' in df.columns:
        df = df.withColumn('repository',
            when(col('repository').rlike(r'^\[.*\]$'),
                 trim(regexp_replace(
                     regexp_extract(col('repository'), r'^\[(.*?),.*\]$|^\[(.*?)\]$', 1),
                     r'^["\']|["\']$', ''
                 ))
            ).otherwise(col('repository')))
    
    # Filtra datas válidas
    if 'commit_time_utc' in df.columns:
        df = df.filter(col('commit_time_utc').isNotNull())
        df = df.filter((year(col('commit_time_utc')) >= 2000) & (year(col('commit_time_utc')) <= 2024))
    
    # 1. Top autores
    if 'author_name' in df.columns:
        contagem_autores = df.filter(col('author_name').isNotNull()) \
                             .groupBy('author_name').count().orderBy(desc('count'))
        total_autores = contagem_autores.count()
        print(f"\nEncontrados {total_autores} autores únicos")
        print("Top 5 autores por número de commits:")
        top_autores = contagem_autores.limit(5).collect()
        for row in top_autores:
            print(f"{row['author_name']}: {row['count']}")
    
    # 2. Frequência por data
    if 'commit_time_utc' in df.columns:
        df_with_date = df.withColumn('data', to_date(col('commit_time_utc')))
        frequencia_commits = df_with_date.filter(col('data').isNotNull()) \
                                         .groupBy('data').count().orderBy(desc('count'))
        total_datas = frequencia_commits.count()
        if total_datas > 0:
            print(f"\nFrequência de commits calculada para {total_datas} datas")
            print("Frequência de commits por data (top 5):")
            top_datas = frequencia_commits.limit(5).collect()
            for row in top_datas:
                print(f"{row['data']}: {row['count']}")
    
    # 3. Top repositórios
    if 'repository' in df.columns:
        contagem_repos = df.filter(col('repository').isNotNull()) \
                           .filter(col('repository') != '') \
                           .groupBy('repository').count().orderBy(desc('count'))
        total_repos = contagem_repos.count()
        print(f"\nEncontrados {total_repos} repositórios únicos")
        print("Top 5 repositórios por número de commits:")
        top_repos = contagem_repos.limit(5).collect()
        for row in top_repos:
            print(f"{row['repository']}: {row['count']}")
    
    # 4. Tamanho médio mensagens
    if 'commit_message' in df.columns:
        media_tamanho = df.filter(col('commit_message').isNotNull()) \
                         .select(avg(length(col('commit_message'))).alias('avg_length')) \
                         .collect()[0]['avg_length']
        if media_tamanho is not None:
            print(f"\nTamanho médio das mensagens de commit: {media_tamanho:.2f} caracteres")
    
    # Total processado
    total_commits = df.count()
    print(f"\nTotal de commits processados: {total_commits}")

def main():
    parser = argparse.ArgumentParser(description='Processa dados de commits do GitHub usando Spark')
    parser.add_argument('--input', required=True, help='Diretório contendo arquivos (CSV, JSON ou Parquet)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Erro: Diretório não existe: {args.input}")
        return
    
    spark = SparkSession.builder \
        .appName("GitHubCommitsAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    try:
        tempo_inicio = time.time()
        processar_commits_github_spark(spark, args.input)
        tempo_fim = time.time()
        print(f"\nTempo total de processamento: {tempo_fim - tempo_inicio:.2f} segundos")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, year, avg, count, desc, to_date, to_timestamp, when, get_json_object
import os
import argparse
import time

def processar_dados_pypi_spark(spark, diretorio_entrada):
    """Processa dados do PyPI do diretório de entrada usando Spark"""
    
    # Lê todos os arquivos da pasta de uma vez
    if "parquet" in diretorio_entrada or any("parquet" in f for f in os.listdir(diretorio_entrada) if os.path.isfile(os.path.join(diretorio_entrada, f))):
        df = spark.read.parquet(f"{diretorio_entrada}/*.parquet")
        print(f"✅ Lidos arquivos Parquet da pasta: {df.count()} registros")

    elif "json" in diretorio_entrada or any("json" in f for f in os.listdir(diretorio_entrada) if os.path.isfile(os.path.join(diretorio_entrada, f))):
        df = spark.read.json(f"{diretorio_entrada}/*.json")
        print(f"✅ Lidos arquivos JSON da pasta: {df.count()} registros")
        # Converte timestamp para JSON
        if 'timestamp' in df.columns:
            df = df.withColumn("timestamp", 
                              when(col("timestamp").rlike("^\\d+$"), 
                                   to_timestamp(col("timestamp").cast("long")))
                              .otherwise(to_timestamp(col("timestamp"))))

    elif "csv" in diretorio_entrada or any("csv" in f for f in os.listdir(diretorio_entrada) if os.path.isfile(os.path.join(diretorio_entrada, f))):
        df = spark.read.csv(f"{diretorio_entrada}/*.csv", header=True, inferSchema=True)
        print(f"✅ Lidos arquivos CSV da pasta: {df.count()} registros")
        # Converte timestamp para CSV
        if 'timestamp' in df.columns:
            df = df.withColumn("timestamp", to_timestamp(col("timestamp")))

    else:
        print(f"❌ Formato de arquivo não suportado no diretório: {diretorio_entrada}")
        raise "Error with file format"
    
    # Extrai informações do arquivo
    if 'file' in df.columns:
        df = df.withColumn('filename', get_json_object(col('file'), '$.filename')) \
               .withColumn('file_project', get_json_object(col('file'), '$.project')) \
               .withColumn('file_version', get_json_object(col('file'), '$.version')) \
               .withColumn('file_type', get_json_object(col('file'), '$.type'))
        df = df.withColumn('filename', 
                          when(col('filename').isNull() & col('file').isNotNull(), col('file'))
                          .otherwise(col('filename')))
    
    # Filtra timestamps válidos
    if 'timestamp' in df.columns:
        df = df.filter(col('timestamp').isNotNull())
        df = df.filter((year(col('timestamp')) >= 2000) & (year(col('timestamp')) <= 2024))
    
    # 1. Top países
    if 'country_code' in df.columns:
        contagem_paises = df.filter(col('country_code').isNotNull()) \
                           .groupBy('country_code').count().orderBy(desc('count'))
        total_paises = contagem_paises.count()
        print(f"\nEncontrados {total_paises} países únicos")
        print("Top 5 países por número de downloads:")
        top_paises = contagem_paises.limit(5).collect()
        for row in top_paises:
            print(f"{row['country_code']}: {row['count']}")
    
    # 2. Frequência por data
    if 'timestamp' in df.columns:
        df_with_date = df.withColumn('data', to_date(col('timestamp')))
        frequencia_downloads = df_with_date.filter(col('data').isNotNull()) \
                                          .groupBy('data').count().orderBy(desc('count'))
        total_datas = frequencia_downloads.count()
        if total_datas > 0:
            print(f"\nFrequência de downloads calculada para {total_datas} datas")
            print("Frequência de downloads por data (top 5):")
            top_datas = frequencia_downloads.limit(5).collect()
            for row in top_datas:
                print(f"{row['data']}: {row['count']}")
    
    # 3. Top projetos
    if 'project' in df.columns:
        contagem_projetos = df.filter(col('project').isNotNull()) \
                           .filter(col('project') != '') \
                           .groupBy('project').count().orderBy(desc('count'))
        total_projetos = contagem_projetos.count()
        print(f"\nEncontrados {total_projetos} projetos únicos")
        print("Top 5 projetos por número de downloads:")
        top_projetos = contagem_projetos.limit(5).collect()
        for row in top_projetos:
            print(f"{row['project']}: {row['count']}")
    
    # 4. Top arquivos
    if 'filename' in df.columns:
        contagem_arquivos = df.filter(col('filename').isNotNull()) \
                           .filter(col('filename') != '') \
                           .groupBy('filename').count().orderBy(desc('count'))
        total_arquivos = contagem_arquivos.count()
        print(f"\nEncontrados {total_arquivos} arquivos únicos")
        print("Top 5 arquivos por número de downloads:")
        top_arquivos = contagem_arquivos.limit(5).collect()
        for row in top_arquivos:
            print(f"{row['filename']}: {row['count']}")
    
    # 5. Top URLs
    if 'url' in df.columns:
        contagem_urls = df.filter(col('url').isNotNull()) \
                         .filter(col('url') != '') \
                         .groupBy('url').count().orderBy(desc('count'))
        total_urls = contagem_urls.count()
        print(f"\nEncontrados {total_urls} URLs únicas")
        print("Top 5 URLs por número de downloads:")
        top_urls = contagem_urls.limit(5).collect()
        for row in top_urls:
            print(f"{row['url']}: {row['count']}")
    
    # Total processado
    total_downloads = df.count()
    print(f"\nTotal de downloads processados: {total_downloads}")

def main():
    parser = argparse.ArgumentParser(description='Processa dados de downloads do PyPI usando Spark')
    parser.add_argument('--input', required=True, help='Diretório contendo arquivos (CSV, JSON ou Parquet)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Erro: Diretório não existe: {args.input}")
        return
    
    spark = SparkSession.builder \
        .appName("PyPIDownloadsAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    try:
        tempo_inicio = time.time()
        processar_dados_pypi_spark(spark, args.input)
        tempo_fim = time.time()
        print(f"\nTempo total de processamento: {tempo_fim - tempo_inicio:.2f} segundos")
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import argparse

def processar_pasta_github_spark(input_path: str):
    spark = SparkSession.builder.appName("GitHubRepoAnalysis").getOrCreate()

    dfs = []
    for filename in os.listdir(input_path):
        if filename.endswith(".csv"):
            df = spark.read.csv(os.path.join(input_path, filename), header=True, inferSchema=True, multiLine=False)
        elif filename.endswith(".json"):
            df = spark.read.json(os.path.join(input_path, filename), multiLine=False)
        else:
            continue

        # Filtra registros com repo.name não nulo, não booleano e string não vazia
        from pyspark.sql.functions import length
        df_filtered = df.filter(
            (col("`repo.name`").isNotNull()) &
            (length(col("`repo.name`").cast("string")) > 1)
        )
        # Agrupa por repo.name e conta
        df_grouped = df_filtered.groupBy("`repo.name`").count()

        first_row = df_grouped.orderBy(col("count").desc()).first()
        print(f"Top repos no arquivo {filename}: {first_row['repo.name']} with {first_row['count']} commits")


        dfs.append(df_grouped)

    if dfs:
        # União de todos os dataframes
        df_union = dfs[0]
        for df in dfs[1:]:
            df_union = df_union.union(df)

        # Agrupa novamente para somar as contagens
        df_global = df_union.groupBy("`repo.name`").sum("count").withColumnRenamed("sum(count)", "count")
        first_row = df_global.orderBy(col("count").desc()).first()
        print(f"Overall top repo: {first_row['repo.name']} with {first_row['count']} commits")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar arquivos GitHub no Spark")
    parser.add_argument("--input", required=True, help="Caminho da pasta com arquivos CSV e JSON")
    args = parser.parse_args()

    processar_pasta_github_spark(args.input)

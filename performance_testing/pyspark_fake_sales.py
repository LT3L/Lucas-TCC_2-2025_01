#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pyspark_fake_sales.py  –  benchmark em PySpark
Leitura de CSV / JSON-lines / Parquet, execução das consultas
e escrita de resultados em um diretório temporário (apagado ao fim).
"""

import argparse
import shutil
import tempfile
from pathlib import Path
import psutil

from pyspark.sql import SparkSession, functions as F


# ─────────────────────────── utilidades ────────────────────────────────────
def find_file(base_dir: Path, basename: str) -> Path:
    """Encontra a primeira ocorrência customers.csv|json|parquet etc."""
    for ext in (".csv", ".json", ".parquet"):
        p = base_dir / f"{basename}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{basename}.* não encontrado em {base_dir}")


def read_table(spark: SparkSession, base_dir: Path, basename: str):
    path = find_file(base_dir, basename)
    ext  = path.suffix.lower()

    if ext == ".csv":
        return (
            spark.read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .csv(str(path))
        )
    if ext in (".json", ".ndjson"):
        return (
            spark.read
                 .option("multiLine", "false")
                 .json(str(path))
        )
    if ext == ".parquet":
        return spark.read.parquet(str(path))
    raise ValueError(f"Extensão não suportada: {path}")


def export(df, out_path: Path):
    """Tenta gravar em Parquet; cai em CSV se Parquet falhar (sem pandas)."""
    try:
        df.write.mode("overwrite").parquet(str(out_path.with_suffix(".parquet")))
    except Exception:
        df.write.mode("overwrite").csv(str(out_path.with_suffix(".csv")), header=True)


# ─────────────────────────── pipeline ──────────────────────────────────────
def executar_analises(base_dir: Path):
    # calcula 90 % da RAM física → arredonda para inteiro de GB
    mem_total_gb = int(psutil.virtual_memory().total / 1024 ** 3 * 0.90)

    spark = (
        SparkSession.builder
        .appName("FakeSalesBenchmark")
        .master("local[*]")
        .config("spark.driver.memory", f"{mem_total_gb}g")  # heap do driver
        .config("spark.executor.memory", f"{mem_total_gb}g")  # mesmo processo
        .config("spark.memory.fraction", "0.8")  # 80 % do heap vira Execution/Storage
        .config("spark.memory.storageFraction", "0.3")  # 30 % p/ cache (0.3×0.8 ≈ 24 %)
        .config("spark.sql.shuffle.partitions", "8")  # menos partições → menos hashes na RAM
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # leitura das tabelas ----------------------------------------------------------------
    customers     = read_table(spark, base_dir, "customers").cache()
    employees     = read_table(spark, base_dir, "employees").cache()
    payment_types = read_table(spark, base_dir, "payment_types").cache()
    products      = read_table(spark, base_dir, "products").cache()
    stores        = read_table(spark, base_dir, "stores").cache()

    sales = (
        read_table(spark, base_dir, "sales")
        .withColumn("sale_date",  F.col("sale_date").cast("timestamp"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("quantity",     F.col("quantity").cast("int"))
        .cache()
    )

    # consultas --------------------------------------------------------------------------
    vendas_por_loja = (
        sales.groupBy("store_id")
             .agg(F.sum("total_amount").alias("total_sales"))
             .orderBy("store_id")
    )

    top_produtos = (
        sales.join(products, "product_id", "left")
             .groupBy("product_id", "product_name")
             .agg(F.sum("total_amount").alias("total_amount"))
             .orderBy(F.col("total_amount").desc())
             .limit(10)
    )

    ticket_pgto = (
        sales.join(payment_types, "payment_type_id", "left")
             .groupBy("payment_method")
             .agg(F.avg("total_amount").alias("avg_ticket"))
             .orderBy(F.col("avg_ticket").desc())
    )

    vendas_cidade = (
        sales.join(stores, "store_id", "left")
             .groupBy("city")
             .agg(F.sum("total_amount").alias("total_sales"))
             .orderBy(F.col("total_sales").desc())
    )

    faturamento_mensal = (
        sales.withColumn("ym", F.date_trunc("month", "sale_date"))
             .groupBy("ym")
             .agg(F.sum("total_amount").alias("total_sales"))
             .orderBy("ym")
    )

    # saída temporária -------------------------------------------------------------------
    temp_dir = Path(tempfile.mkdtemp(prefix="fake_sales_spark_"))
    export(vendas_por_loja,    temp_dir / "vendas_por_loja")
    export(top_produtos,       temp_dir / "top_produtos")
    export(ticket_pgto,        temp_dir / "ticket_pgto")
    export(vendas_cidade,      temp_dir / "vendas_cidade")
    export(faturamento_mensal, temp_dir / "faturamento_mensal")

    print(f"Resultados salvos em {temp_dir} (apagados ao encerrar)")

    # encerra Spark e remove diretório ---------------------------------------------------
    spark.stop()
    shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────── CLI ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PySpark – Fake Sales")
    parser.add_argument("--input", required=True, help="Pasta com os datasets")
    args = parser.parse_args()

    executar_analises(Path(args.input))
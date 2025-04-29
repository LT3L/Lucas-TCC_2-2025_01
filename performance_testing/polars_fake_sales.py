#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
polars_fake_sales.py  –  benchmark em Polars
Leitura de CSV / JSON (lines) / Parquet e grava
resultados em arquivos temporários que são apagados ao fim.
"""

import argparse
import tempfile
from pathlib import Path
import polars as pl

# ────────────────────────────────────────────────────────────────────────────
# localizar arquivo + ler em Polars
# ────────────────────────────────────────────────────────────────────────────
def find_file(base_dir: Path, basename: str) -> Path:
    for ext in (".csv", ".json", ".parquet"):
        p = base_dir / f"{basename}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{basename}.* não encontrado em {base_dir}")

def read_table(base_dir: Path, basename: str) -> pl.DataFrame:
    path = find_file(base_dir, basename)
    ext  = path.suffix.lower()

    if ext == ".csv":
        return pl.read_csv(path, low_memory=True)
    if ext in (".json", ".ndjson"):
        return pl.read_ndjson(path)
    if ext == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Extensão não suportada: {path}")

def export(df: pl.DataFrame, out_path: Path):
    try:
        df.write_parquet(out_path.with_suffix(".parquet"))
    except Exception:
        df.write_csv(out_path.with_suffix(".csv"))

# ────────────────────────────────────────────────────────────────────────────
# pipeline principal
# ────────────────────────────────────────────────────────────────────────────
def executar_analises(base_dir: Path):

    customers     = read_table(base_dir, "customers")
    employees     = read_table(base_dir, "employees")
    payment_types = read_table(base_dir, "payment_types")
    products      = read_table(base_dir, "products")
    stores        = read_table(base_dir, "stores")

    sales = read_table(base_dir, "sales")

    # ajustes de tipos
    sales = (
        sales
        .with_columns([
            pl.col("sale_date").str.to_datetime(time_unit="ms", time_zone="UTC"),
            pl.col("total_amount").cast(pl.Float32),
            pl.col("quantity").cast(pl.Int32),
        ])
    )

    # 2.1 total de vendas por loja
    vendas_por_loja = (
        sales
        .group_by("store_id")
        .agg(pl.col("total_amount").sum().alias("total_sales"))
        .sort("store_id")
    )

    # 2.2 top-10 produtos por receita
    top_produtos = (
        sales.join(products, on="product_id", how="left")
             .group_by("product_id", "product_name")
             .agg(pl.col("total_amount").sum())
             .sort("total_amount", descending=True)
             .head(10)
    )

    # 2.3 ticket médio por tipo de pagamento
    ticket_pgto = (
        sales.join(payment_types, on="payment_type_id", how="left")
             .group_by("payment_method")
             .agg(pl.col("total_amount").mean().alias("avg_ticket"))
             .sort("avg_ticket", descending=True)
    )

    # 2.4 vendas por cidade
    vendas_cidade = (
        sales.join(stores, on="store_id", how="left")
             .group_by("city")
             .agg(pl.col("total_amount").sum().alias("total_sales"))
             .sort("total_sales", descending=True)
    )

    # 2.5 faturamento mensal (Year-Month)
    faturamento_mensal = (
        sales
        .with_columns(pl.col("sale_date").dt.truncate("1mo").alias("ym"))
        .group_by("ym")
        .agg(pl.col("total_amount").sum().alias("total_sales"))
        .sort("ym")
    )

    # 3) salva resultados numa pasta temporária
    with tempfile.TemporaryDirectory(prefix="fake_sales_polars_") as tmp:
        out_dir = Path(tmp)
        export(vendas_por_loja,    out_dir / "vendas_por_loja")
        export(top_produtos,       out_dir / "top_produtos")
        export(ticket_pgto,        out_dir / "ticket_pgto")
        export(vendas_cidade,      out_dir / "vendas_cidade")
        export(faturamento_mensal, out_dir / "faturamento_mensal")

        print(f"Resultados salvos em {out_dir} (apagados ao encerrar)")

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Polars – Fake Sales")
    parser.add_argument("--input", required=True, help="Pasta com os datasets")
    args = parser.parse_args()

    executar_analises(Path(args.input))
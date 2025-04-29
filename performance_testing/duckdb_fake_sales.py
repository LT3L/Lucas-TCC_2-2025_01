#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
duckdb_fake_sales.py  –  benchmark DuckDB
* Lê CSV / JSON-lines / Parquet.
* Executa 5 consultas.
* Salva cada resultado em Parquet (ou, se falhar, em CSV) via COPY do próprio DuckDB.
* Todos os arquivos ficam em um diretório temporário que é removido ao final.
"""

import argparse
import tempfile
from pathlib import Path
import duckdb


# ─────────────────────────── utilidades ────────────────────────────────────
def find_file(base_dir: Path, basename: str) -> Path:
    for ext in (".csv", ".json", ".parquet"):
        p = base_dir / f"{basename}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{basename}.* não encontrado em {base_dir}")


def register_table(conn: duckdb.DuckDBPyConnection, name: str, path: Path):
    ext = path.suffix.lower()
    if ext == ".csv":
        conn.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}');")
    elif ext in (".json", ".ndjson"):
        conn.execute(f"CREATE VIEW {name} AS SELECT * FROM read_ndjson_auto('{path}');")
    elif ext == ".parquet":
        conn.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}');")
    else:
        raise ValueError(f"Extensão não suportada: {path}")


def export_relation(conn: duckdb.DuckDBPyConnection, sql: str, out: Path):
    """Tenta escrever Parquet; se falhar, salva CSV (ainda via DuckDB)."""
    try:
        conn.execute(f"COPY ({sql}) TO '{out.with_suffix('.parquet')}' (FORMAT 'parquet');")
    except Exception:
        conn.execute(f"COPY ({sql}) TO '{out.with_suffix('.csv')}' (HEADER, DELIMITER ',');")


# ─────────────────────────── pipeline ──────────────────────────────────────
def executar_analises(base_dir: Path):
    conn = duckdb.connect(database=":memory:")

    # registra as 6 tabelas
    for tbl in ("customers", "employees", "payment_types", "products", "sales", "stores"):
        register_table(conn, tbl, find_file(base_dir, tbl))

    # diretório temporário para resultados
    with tempfile.TemporaryDirectory(prefix="fake_sales_duckdb_") as tmp:
        out = Path(tmp)

        export_relation(conn, """
            SELECT store_id, SUM(total_amount) AS total_sales
            FROM sales GROUP BY store_id ORDER BY store_id
        """, out / "vendas_por_loja")

        export_relation(conn, """
            SELECT p.product_id, p.product_name, SUM(s.total_amount) AS total_amount
            FROM sales AS s
            LEFT JOIN products AS p USING (product_id)
            GROUP BY p.product_id, p.product_name
            ORDER BY total_amount DESC LIMIT 10
        """, out / "top_produtos")

        export_relation(conn, """
            SELECT pt.payment_method, AVG(s.total_amount) AS avg_ticket
            FROM sales AS s
            LEFT JOIN payment_types AS pt USING (payment_type_id)
            GROUP BY pt.payment_method
            ORDER BY avg_ticket DESC
        """, out / "ticket_pgto")

        export_relation(conn, """
            SELECT st.city, SUM(s.total_amount) AS total_sales
            FROM sales AS s
            LEFT JOIN stores AS st USING (store_id)
            GROUP BY st.city
            ORDER BY total_sales DESC
        """, out / "vendas_cidade")

        export_relation(conn, """
            SELECT date_trunc('month', sale_date) AS ym,
                   SUM(total_amount) AS total_sales
            FROM sales
            GROUP BY ym
            ORDER BY ym
        """, out / "faturamento_mensal")

        print(f"Resultados salvos em {out} (apagados ao encerrar)")

    conn.close()


# ─────────────────────────── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DuckDB – Fake Sales")
    parser.add_argument("--input", required=True, help="Pasta com os datasets")
    args = parser.parse_args()

    executar_analises(Path(args.input))
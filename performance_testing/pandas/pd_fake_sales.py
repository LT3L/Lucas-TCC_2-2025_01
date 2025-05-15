"""
pd_fake_sales.py  –  aceita CSV, JSON (lines) ou Parquet
"""

import argparse
from pathlib import Path
import tempfile
import pandas as pd


def find_file(base_dir: Path, basename: str) -> Path:
    for ext in (".csv", ".json", ".parquet"):
        candidate = base_dir / f"{basename}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Nenhum arquivo {basename} com extensão CSV/JSON/Parquet encontrado em {base_dir}"
    )

def read_table(base_dir: Path, basename: str, **kwargs) -> pd.DataFrame:
    path = find_file(base_dir, basename)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path, low_memory=False, dtype=str, **kwargs)
    if ext in (".json", ".ndjson"):
        return pd.read_json(path, dtype=str, lines=True, **kwargs)
    if ext == ".parquet":
        return pd.read_parquet(path, **kwargs).astype(str)
    raise ValueError(f"Extensão não suportada: {path}")


def export(df: pd.DataFrame, out_path: Path):
    """Salva em Parquet; se não houver pyarrow cai para CSV."""
    try:
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(out_path.with_suffix(".csv"), index=False)

# --------------------------------------------------------------------------- #
# pipeline                                                                    #
# --------------------------------------------------------------------------- #
def executar_analises(base_dir: Path):
    # 1) Lê tabelas (qualquer formato)
    customers     = read_table(base_dir, "customers")
    employees     = read_table(base_dir, "employees")
    payment_types = read_table(base_dir, "payment_types")
    products      = read_table(base_dir, "products")
    stores        = read_table(base_dir, "stores")

    sales = read_table(base_dir, "sales")  # sem parse_dates aqui
    sales["sale_date"] = pd.to_datetime(
        sales["sale_date"],
        errors="coerce",  # vira NaT se falhar
        utc=True
    )

    # Ajusta dtypes numéricos
    sales["total_amount"] = sales["total_amount"].astype("float32")
    sales["quantity"]     = sales["quantity"].astype("int32")

    # 2) Métricas -------------------------------------------------------------
    # 2.1 total de vendas por loja
    vendas_por_loja = (
        sales.groupby("store_id")["total_amount"]
             .sum()
             .reset_index(name="total_sales")
    )

    # 2.2 top-10 produtos por receita
    top_produtos = (
        sales.merge(products, on="product_id", how="left")
             .groupby(["product_id", "product_name"])["total_amount"]
             .sum()
             .nlargest(10)
             .reset_index()
    )

    # 2.3 ticket médio por tipo de pagamento
    ticket_pgto = (
        sales.merge(payment_types, on="payment_type_id", how="left")
             .groupby("payment_method")["total_amount"]
             .mean()
             .reset_index(name="avg_ticket")
    )

    # 2.4 vendas por cidade
    vendas_cidade = (
        sales.merge(stores, on="store_id", how="left")
             .groupby("city")["total_amount"]
             .sum()
             .reset_index(name="total_sales")
    )

    # 2.5 faturamento mensal
    faturamento_mensal = (
        sales
        .assign(
            ym=(
                sales["sale_date"]
                .dt.tz_localize(None)  # descarta o timezone
                .dt.to_period("M")  # converte para ano-mês
            )
        )
        .groupby("ym")["total_amount"]
        .sum()
        .reset_index(name="total_sales")
    )

    # 3) imprime resultados
    with tempfile.TemporaryDirectory(prefix="fake_sales_") as tmp:
        out_dir = Path(tmp)
        export(vendas_por_loja,    out_dir / "vendas_por_loja")
        export(top_produtos,       out_dir / "top_produtos")
        export(ticket_pgto,        out_dir / "ticket_pgto")
        export(vendas_cidade,      out_dir / "vendas_cidade")
        export(faturamento_mensal, out_dir / "faturamento_mensal")

        # deixa apenas um aviso curto; a main mede tempo/memória
        print(f"Resultados salvos em {out_dir} (serão apagados ao encerrar)")



# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análises Pandas (Fake Sales)")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Pasta contendo os arquivos (csv / json / parquet)",
    )
    args = parser.parse_args()

    executar_analises(Path(args.input))
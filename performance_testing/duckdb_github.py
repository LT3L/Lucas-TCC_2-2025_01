import argparse
import duckdb
import os
import polars as pl

def processar_pasta_github_polars(pasta):
    dfs_parciais = []
    for arquivo in os.listdir(pasta):
        caminho_arquivo = os.path.join(pasta, arquivo)
        if arquivo.endswith('.csv'):
            df = pl.read_csv(caminho_arquivo)
            counts = df.group_by("repo.name").agg(pl.len().alias("count"))
            dfs_parciais.append(counts)
        elif arquivo.endswith('.json'):
            query = f'SELECT "repo.name", COUNT(*) as count FROM read_ndjson_auto(\'{caminho_arquivo}\') GROUP BY "repo.name"'
            df_json = duckdb.query(query).fetchdf()
            counts = pl.from_pandas(df_json)
            dfs_parciais.append(counts)

    if dfs_parciais:
        resultado_global = pl.concat(dfs_parciais).group_by("repo.name").agg(pl.sum("count").alias("count")).sort("count", descending=True)
        top_repo = resultado_global.row(0)
        print(f"Overall top repo: {top_repo[0]} with {top_repo[1]} commits")
    else:
        print("No commits found in any CSV or JSON files.")

def main():
    parser = argparse.ArgumentParser(description="Process CSV files to count commits per repo.")
    parser.add_argument('--input', required=True, help='Path to the folder containing CSV files')
    args = parser.parse_args()

    processar_pasta_github_polars(args.input)

if __name__ == "__main__":
    main()

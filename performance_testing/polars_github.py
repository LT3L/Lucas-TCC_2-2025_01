import polars as pl
import os
import sys
import argparse

def processar_pasta_github_polars(path: str):
    """
    Processa arquivos CSV e JSON de commits do GitHub usando Polars.
    Conta o número de commits por repo.name e mostra o repositório com mais commits por arquivo e globalmente.
    """
    arquivos = [f for f in os.listdir(path) if f.endswith(".csv") or f.endswith(".json")]
    arquivos.sort()

    print(f"📁 Processando {len(arquivos)} arquivos com Polars na pasta {path}...\n")

    dfs_contagem = []

    print("Top repositórios por arquivo:")
    for nome_arquivo in arquivos:
        caminho = os.path.join(path, nome_arquivo)

        try:
            if nome_arquivo.endswith(".csv"):
                df = pl.read_csv(caminho)
            else:
                df = pl.read_ndjson(caminho, infer_schema_length=1000)

            if "repo.name" not in df.columns:
                print(f"{nome_arquivo}: coluna 'repo.name' não encontrada.")
                continue

            counts = (
                df.group_by("repo.name")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )

            if counts.shape[0] == 0:
                print(f"{nome_arquivo}: sem dados em 'repo.name'")
                continue

            top_repo = counts[0, "repo.name"]
            top_count = counts[0, "count"]
            print(f"{nome_arquivo}: {top_repo} com {top_count} commits")

            dfs_contagem.append(counts)

        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {e}")

    if dfs_contagem:
        resultado_global = (
            pl.concat(dfs_contagem)
            .group_by("repo.name")
            .agg(pl.sum("count").alias("count"))
            .sort("count", descending=True)
        )
        top_global = resultado_global.row(0)
        repo_global = top_global[0]
        total_commits = top_global[1]
        print(f"\n🏆 Top repositório global: {repo_global} com {total_commits} commits")
    else:
        print("\n⚠️ Nenhum dado válido encontrado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa arquivos de commits do GitHub usando Polars.")
    parser.add_argument('--input', required=True, help='Caminho da pasta contendo os arquivos a serem processados')
    args = parser.parse_args()
    processar_pasta_github_polars(args.input)
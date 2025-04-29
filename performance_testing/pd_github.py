import os
import pandas as pd
import json

def processar_pasta_github(path: str):
    top_repos_por_arquivo = {}
    dfs_contagem = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.json'):
                file_path = os.path.join(root, file)
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path, low_memory=False)
                else:
                    # Lê arquivos .json no formato NDJSON (JSON Lines)
                    df = pd.read_json(file_path, lines=True)

                if 'repo.name' in df.columns:
                    commits_por_repo = df['repo.name'].value_counts()
                    top_repo = commits_por_repo.idxmax()
                    top_repos_por_arquivo[file] = (top_repo, commits_por_repo.max())

                    df_contagem = commits_por_repo.reset_index()
                    df_contagem.columns = ['repo.name', 'count']
                    dfs_contagem.append(df_contagem)

    if dfs_contagem:
        df_global = pd.concat(dfs_contagem).groupby('repo.name', as_index=False)['count'].sum()
        top_global_row = df_global.sort_values('count', ascending=False).iloc[0]
        top_global_repo = top_global_row['repo.name']
        top_global_count = top_global_row['count']
    else:
        top_global_repo = None
        top_global_count = 0

    print("Top repositórios por arquivo:")
    for file, (repo, count) in top_repos_por_arquivo.items():
        print(f"{file}: {repo} com {count} commits")

    if top_global_repo:
        print(f"\nTop repositório global: {top_global_repo} com {top_global_count} commits")
    else:
        print("\nNenhum dado de commits encontrado.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Caminho da pasta com arquivos GitHub')
    args = parser.parse_args()

    processar_pasta_github(args.input)


if __name__ == "__main__":
    main()
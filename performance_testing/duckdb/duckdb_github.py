import duckdb
import argparse
import os
from glob import glob
import time
from datetime import datetime

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Script DuckDB GitHub")
parser.add_argument('--input', type=str, required=True, help='Caminho para o diretório de entrada')
args = parser.parse_args()

diretorio = args.input

def detectar_formato_arquivo(diretorio):
    """Detecta o formato do arquivo baseado na extensão do primeiro arquivo encontrado"""
    # Procura por arquivos em ordem: csv, json, parquet
    for ext in ['.csv', '.json', '.parquet']:
        arquivos = glob(os.path.join(diretorio, f'*{ext}'))
        if arquivos:
            return ext.replace('.', '')
    return None

def ler_arquivo(caminho, formato):
    """Lê o arquivo no formato especificado"""
    if formato == 'csv':
        df = duckdb.sql(f"SELECT * FROM read_csv_auto('{caminho}/*.csv')").df()
    elif formato == 'json':
        df = duckdb.sql(f"SELECT * FROM read_json_auto('{caminho}/*.json')").df()
    elif formato == 'parquet':
        df = duckdb.sql(f"SELECT * FROM '{caminho}/*.parquet'").df()
    else:
        raise ValueError(f"Formato não suportado: {formato}")
    
    return df

def processar_commits_github(diretorio_entrada):
    """Processa dados de commits do GitHub do diretório de entrada"""
    # Detecta o formato e lê os arquivos
    formato = detectar_formato_arquivo(diretorio_entrada)
    if not formato:
        print(f"Nenhum arquivo válido encontrado em {diretorio_entrada}")
        return

    print(f"Formato detectado: {formato}")
    df = ler_arquivo(diretorio_entrada, formato)

    # Criar conexão para processamento SQL
    con = duckdb.connect()
    con.register("commits", df)

    # Processamento SQL (filtros e colunas derivadas)
    query = """
    SELECT *,
           LENGTH(commit_message) as message_length,
           EXTRACT(YEAR FROM to_timestamp(commit_time_utc)) as year,
           EXTRACT(MONTH FROM to_timestamp(commit_time_utc)) as month,
           EXTRACT(DAY FROM to_timestamp(commit_time_utc)) as day,
           EXTRACT(HOUR FROM to_timestamp(commit_time_utc)) as hour
    FROM commits
    WHERE commit_message IS NOT NULL 
      AND author_name IS NOT NULL
      AND LENGTH(commit_message) > 0
      AND LENGTH(author_name) > 0
      AND to_timestamp(commit_time_utc) >= '2000-01-01'
      AND to_timestamp(commit_time_utc) <= CURRENT_TIMESTAMP
    """
    df = con.execute(query).df()

    # 1. Contagem de commits por autor
    if 'author_name' in df.columns:
        query = """
        SELECT author_name, COUNT(*) as commit_count
        FROM commits
        GROUP BY author_name
        ORDER BY commit_count DESC
        LIMIT 5
        """
        contagem_autores = con.execute(query).df()
        print(f"\nEncontrados {len(contagem_autores)} autores únicos")
        print("Top 5 autores por número de commits:")
        print(contagem_autores)

    # 2. Frequência de commits ao longo do tempo
    if 'commit_time_utc' in df.columns:
        query = """
        SELECT CAST(to_timestamp(commit_time_utc) AS DATE) as data,
               COUNT(*) as commit_count
        FROM commits
        GROUP BY data
        ORDER BY commit_count DESC
        LIMIT 5
        """
        frequencia_commits = con.execute(query).df()
        print(f"\nFrequência de commits calculada para {len(frequencia_commits)} datas")
        print("Frequência de commits por data (top 5):")
        print(frequencia_commits)

    # 3. Estatísticas de repositórios
    if 'repository' in df.columns:
        query = """
        SELECT repository, COUNT(*) as commit_count
        FROM commits
        GROUP BY repository
        ORDER BY commit_count DESC
        LIMIT 5
        """
        contagem_repos = con.execute(query).df()
        print(f"\nEncontrados {len(contagem_repos)} repositórios únicos")
        print("Top 5 repositórios por número de commits:")
        print(contagem_repos)

    # 4. Estatísticas de mensagens de commit
    if 'commit_message' in df.columns:
        query = """
        SELECT AVG(LENGTH(commit_message)) as media_tamanho_msg
        FROM commits
        """
        media_tamanho_msg = con.execute(query).df().iloc[0, 0]
        print(f"\nTamanho médio das mensagens de commit: {media_tamanho_msg:.2f} caracteres")

    # Total de commits
    query = "SELECT COUNT(*) as total_commits FROM commits"
    total_commits = con.execute(query).df().iloc[0, 0]
    print(f"\nTotal de commits processados: {total_commits}")

def main():
    # Verifica se o diretório existe
    if not os.path.exists(args.input):
        print(f"Erro: Diretório não existe: {args.input}")
        return
    
    # Processa os arquivos
    tempo_inicio = time.time()
    processar_commits_github(args.input)
    tempo_fim = time.time()
    
    print(f"\nTempo total de processamento: {tempo_fim - tempo_inicio:.2f} segundos")

if __name__ == "__main__":
    main()
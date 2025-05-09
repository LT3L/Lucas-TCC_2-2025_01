import pandas as pd
import argparse
import os
from glob import glob
import time
import pyarrow.parquet as pq
from datetime import datetime
import numpy as np

def detectar_formato_arquivo(caminho):
    """Detecta o formato do arquivo baseado na extensão"""
    extensao = os.path.splitext(caminho)[1].lower()
    if extensao == '.csv':
        return 'csv'
    elif extensao == '.json':
        return 'json'
    elif extensao == '.parquet':
        return 'parquet'
    return None

def processar_repository(x):
    """Processa o campo repository para garantir que seja uma string"""
    if isinstance(x, (list, np.ndarray)):
        if len(x) > 0:
            return str(x[0])
    return str(x) if x is not None else None

def converter_timestamp(df):
    """Converte o campo commit_time_utc para datetime"""
    if 'commit_time_utc' in df.columns:
        try:
            # Tenta converter como Unix timestamp (segundos)
            df['commit_time_utc'] = pd.to_datetime(df['commit_time_utc'], unit='s', utc=True)
        except Exception:
            try:
                # Tenta converter como string ISO
                df['commit_time_utc'] = pd.to_datetime(df['commit_time_utc'], utc=True)
            except Exception:
                try:
                    # Tenta converter como string com timezone
                    df['commit_time_utc'] = pd.to_datetime(df['commit_time_utc'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
                except Exception:
                    try:
                        # Tenta converter como string sem timezone
                        df['commit_time_utc'] = pd.to_datetime(df['commit_time_utc'], format='%Y-%m-%d %H:%M:%S', utc=True)
                    except Exception:
                        pass
    return df

def ler_arquivo(caminho, formato):
    """Lê o arquivo no formato especificado"""
    if formato == 'csv':
        df = pd.read_csv(caminho)
    elif formato == 'json':
        df = pd.read_json(caminho, lines=True)
    elif formato == 'parquet':
        df = pd.read_parquet(caminho)
    else:
        raise ValueError(f"Formato não suportado: {formato}")
    
    # Converte o timestamp para datetime
    df = converter_timestamp(df)
    
    # Converte a lista de repositório para string
    if 'repository' in df.columns:
        df['repository'] = df['repository'].apply(processar_repository)
    
    return df

def processar_commits_github(diretorio_entrada):
    """Processa dados de commits do GitHub do diretório de entrada"""
    # Obtém todos os arquivos no diretório
    arquivos = []
    for formato in ['*.csv', '*.json', '*.parquet']:
        arquivos.extend(glob(os.path.join(diretorio_entrada, formato)))
    
    if not arquivos:
        print(f"Nenhum arquivo encontrado em {diretorio_entrada}")
        return
    
    # Detecta o formato do primeiro arquivo
    formato = detectar_formato_arquivo(arquivos[0])
    if not formato:
        print(f"Formato não suportado para o arquivo: {arquivos[0]}")
        return
    
    # Lê todos os arquivos
    dfs = []
    for arquivo in sorted(arquivos):
        try:
            df = ler_arquivo(arquivo, formato)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
    
    if not dfs:
        print("Nenhum dado válido encontrado nos arquivos")
        return
    
    # Combina todos os DataFrames
    df = pd.concat(dfs, ignore_index=True)
    
    # Remove linhas com datas inválidas
    df = df.dropna(subset=['commit_time_utc'])
    
    # Remove datas muito antigas (antes de 2000) ou futuras
    if pd.api.types.is_datetime64_any_dtype(df['commit_time_utc']):
        df = df[
            (df['commit_time_utc'].dt.year >= 2000) & 
            (df['commit_time_utc'].dt.year <= pd.Timestamp.now().year)
        ]
    
    # 1. Contagem de commits por autor
    if 'author_name' in df.columns:
        contagem_autores = df['author_name'].value_counts()
        print(f"\nEncontrados {len(contagem_autores)} autores únicos")
        print("Top 5 autores por número de commits:")
        print(contagem_autores.head())
    
    # 2. Frequência de commits ao longo do tempo
    if 'commit_time_utc' in df.columns and not df.empty and pd.api.types.is_datetime64_any_dtype(df['commit_time_utc']):
        df['data'] = df['commit_time_utc'].dt.date
        frequencia_commits = df.groupby('data').size()
        
        if not frequencia_commits.empty:
            print(f"\nFrequência de commits calculada para {len(frequencia_commits)} datas")
            print("Frequência de commits por data (top 5):")
            print(frequencia_commits.sort_values(ascending=False).head())
    
    # 3. Estatísticas de repositórios
    if 'repository' in df.columns:
        contagem_repos = df['repository'].value_counts()
        print(f"\nEncontrados {len(contagem_repos)} repositórios únicos")
        print("Top 5 repositórios por número de commits:")
        print(contagem_repos.head())
    
    # 4. Estatísticas de mensagens de commit
    if 'commit_message' in df.columns:
        media_tamanho_msg = df['commit_message'].str.len().mean()
        print(f"\nTamanho médio das mensagens de commit: {media_tamanho_msg:.2f} caracteres")
    
    print(f"\nTotal de commits processados: {len(df)}")

def main():
    parser = argparse.ArgumentParser(description='Processa dados de commits do GitHub')
    parser.add_argument('--input', required=True, help='Diretório contendo arquivos (CSV, JSON ou Parquet)')
    args = parser.parse_args()
    
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
import pandas as pd
import argparse
import os
from glob import glob
import time
import pyarrow.parquet as pq
from datetime import datetime
import numpy as np
import traceback

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

def converter_timestamp(df, formato):
    """Converte o campo timestamp para datetime baseado no formato do arquivo"""
    if 'timestamp' in df.columns:
        try:
            if formato == 'parquet':
                # Parquet já vem como datetime64[us]
                return df
            elif formato == 'json':
                # JSON vem como datetime64[ns]
                return df
            elif formato == 'csv':
                # CSV vem como string, precisa converter
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Erro ao converter timestamp: {e}")
            return df
    return df

def extrair_info_arquivo(file_dict):
    """Extrai informações do dicionário de arquivo"""
    if isinstance(file_dict, dict):
        return {
            'filename': file_dict.get('filename', ''),
            'project': file_dict.get('project', ''),
            'version': file_dict.get('version', ''),
            'type': file_dict.get('type', '')
        }
    return {'filename': '', 'project': '', 'version': '', 'type': ''}

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
    df = converter_timestamp(df, formato)
    
    # Extrai informações do campo file
    if 'file' in df.columns:
        file_info = df['file'].apply(extrair_info_arquivo)
        df['filename'] = file_info.apply(lambda x: x['filename'])
        df['file_project'] = file_info.apply(lambda x: x['project'])
        df['file_version'] = file_info.apply(lambda x: x['version'])
        df['file_type'] = file_info.apply(lambda x: x['type'])
    
    return df

def processar_dados_pypi(diretorio_entrada):
    """Processa dados do PyPI do diretório de entrada"""
    try:
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
        
        # Remove linhas com timestamps inválidos
        df = df.dropna(subset=['timestamp'])
        
        # Remove datas muito antigas (antes de 2000) ou futuras
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df = df[
                (df['timestamp'].dt.year >= 2000) & 
                (df['timestamp'].dt.year <= pd.Timestamp.now().year)
            ]
        
        # 1. Estatísticas por país
        if 'country_code' in df.columns:
            contagem_paises = df['country_code'].value_counts()
            print(f"\nEncontrados {len(contagem_paises)} países únicos")
            print("Top 5 países por número de downloads:")
            print(contagem_paises.head())
        
        # 2. Frequência de downloads ao longo do tempo
        if 'timestamp' in df.columns and not df.empty and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['data'] = df['timestamp'].dt.date
            frequencia_downloads = df.groupby('data').size()
            
            if not frequencia_downloads.empty:
                print(f"\nFrequência de downloads calculada para {len(frequencia_downloads)} datas")
                print("Frequência de downloads por data (top 5):")
                print(frequencia_downloads.sort_values(ascending=False).head())
        
        # 3. Estatísticas de projetos
        if 'project' in df.columns:
            contagem_projetos = df['project'].value_counts()
            print(f"\nEncontrados {len(contagem_projetos)} projetos únicos")
            print("Top 5 projetos por número de downloads:")
            print(contagem_projetos.head())
        
        # 4. Estatísticas de arquivos
        if 'filename' in df.columns:
            contagem_arquivos = df['filename'].value_counts()
            print(f"\nEncontrados {len(contagem_arquivos)} arquivos únicos")
            print("Top 5 arquivos por número de downloads:")
            print(contagem_arquivos.head())
        
        # 5. Estatísticas de URLs
        if 'url' in df.columns:
            contagem_urls = df['url'].value_counts()
            print(f"\nEncontrados {len(contagem_urls)} URLs únicas")
            print("Top 5 URLs por número de downloads:")
            print(contagem_urls.head())
        
        print(f"\nTotal de downloads processados: {len(df)}")
        
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Processa dados de downloads do PyPI')
    parser.add_argument('--input', required=True, help='Diretório contendo arquivos (CSV, JSON ou Parquet)')
    args = parser.parse_args()

    # Verifica se o diretório existe
    if not os.path.exists(args.input):
        print(f"Erro: Diretório não existe: {args.input}")
        return
    
    # Processa os arquivos
    tempo_inicio = time.time()
    processar_dados_pypi(args.input)
    tempo_fim = time.time()
    
    print(f"\nTempo total de processamento: {tempo_fim - tempo_inicio:.2f} segundos")

if __name__ == "__main__":
    main()
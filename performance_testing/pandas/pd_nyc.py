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

def converter_timestamp(df):
    """Converte os campos de timestamp para datetime"""
    try:
        # Converte pickup e dropoff datetime
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        return df
    except Exception as e:
        print(f"Erro ao converter timestamps: {e}")
        return df

def ler_arquivos(diretorio, formato):
    """Lê todos os arquivos do diretório de uma vez"""
    try:
        if formato == 'csv':
            # Lê todos os CSVs de uma vez
            df = pd.concat([pd.read_csv(f) for f in sorted(glob(os.path.join(diretorio, '*.csv')))], ignore_index=True)
        elif formato == 'json':
            # Lê todos os JSONs de uma vez
            df = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob(os.path.join(diretorio, '*.json')))], ignore_index=True)
        elif formato == 'parquet':
            # Lê todos os Parquets de uma vez
            df = pd.concat([pd.read_parquet(f) for f in sorted(glob(os.path.join(diretorio, '*.parquet')))], ignore_index=True)
        else:
            raise ValueError(f"Formato não suportado: {formato}")
        
        # Converte os timestamps para datetime
        df = converter_timestamp(df)
        
        return df
    except Exception as e:
        print(f"Erro ao ler arquivos: {e}")
        return None

def processar_dados_nyc(diretorio_entrada):
    """Processa dados do NYC do diretório de entrada"""
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
        
        # Lê todos os arquivos de uma vez
        print(f"Lendo todos os arquivos {formato}...")
        df = ler_arquivos(diretorio_entrada, formato)
        
        if df is None or df.empty:
            print("Nenhum dado válido encontrado nos arquivos")
            return
        
        print(f"Total de registros lidos: {len(df)}")
        
        # Remove linhas com timestamps inválidos
        df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        print(f"Registros após remover timestamps inválidos: {len(df)}")
        
        # Remove datas muito antigas (antes de 2000) ou futuras
        df = df[
            (df['tpep_pickup_datetime'].dt.year >= 2000) & 
            (df['tpep_pickup_datetime'].dt.year <= pd.Timestamp.now().year)
        ]
        print(f"Registros após filtrar datas: {len(df)}")
        
        # 1. Estatísticas de passageiros
        if 'passenger_count' in df.columns:
            contagem_passageiros = df['passenger_count'].value_counts()
            print(f"\nDistribuição de passageiros por viagem:")
            print(contagem_passageiros)
        
        # 2. Frequência de viagens ao longo do tempo
        df['data'] = df['tpep_pickup_datetime'].dt.date
        frequencia_viagens = df.groupby('data').size()
        
        if not frequencia_viagens.empty:
            print(f"\nFrequência de viagens calculada para {len(frequencia_viagens)} datas")
            print("Frequência de viagens por data (top 5):")
            print(frequencia_viagens.sort_values(ascending=False).head())
        
        # 3. Estatísticas de distância
        if 'trip_distance' in df.columns:
            print(f"\nEstatísticas de distância das viagens:")
            print(df['trip_distance'].describe())
        
        # 4. Estatísticas de pagamento
        if 'payment_type' in df.columns:
            contagem_pagamentos = df['payment_type'].value_counts()
            print(f"\nDistribuição de tipos de pagamento:")
            print(contagem_pagamentos)
        
        # 5. Estatísticas de valores
        colunas_valores = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount']
        for coluna in colunas_valores:
            if coluna in df.columns:
                print(f"\nEstatísticas de {coluna}:")
                print(df[coluna].describe())
        
        # 6. Estatísticas de localização
        if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
            print("\nTop 5 locais de origem mais frequentes:")
            print(df['PULocationID'].value_counts().head())
            print("\nTop 5 locais de destino mais frequentes:")
            print(df['DOLocationID'].value_counts().head())
        
        print(f"\nTotal de viagens processadas: {len(df)}")
        
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Processa dados de viagens de táxi de NYC')
    parser.add_argument('--input', required=True, help='Diretório contendo arquivos (CSV, JSON ou Parquet)')
    args = parser.parse_args()

    # Verifica se o diretório existe
    if not os.path.exists(args.input):
        print(f"Erro: Diretório não existe: {args.input}")
        return
    
    # Processa os arquivos
    tempo_inicio = time.time()
    processar_dados_nyc(args.input)
    tempo_fim = time.time()
    
    print(f"\nTempo total de processamento: {tempo_fim - tempo_inicio:.2f} segundos")

if __name__ == "__main__":
    main()
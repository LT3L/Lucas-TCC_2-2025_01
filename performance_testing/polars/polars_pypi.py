import polars as pl
import argparse
import os
from glob import glob
import time
import traceback
from datetime import datetime

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
                # Parquet já vem como datetime
                return df
            elif formato == 'json':
                # JSON vem como datetime
                return df
            elif formato == 'csv':
                # CSV vem como string, precisa converter
                df = df.with_columns(pl.col("timestamp").str.to_datetime())
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
        df = pl.read_csv(caminho)
    elif formato == 'json':
        df = pl.read_ndjson(caminho)
    elif formato == 'parquet':
        df = pl.read_parquet(caminho)
    else:
        raise ValueError(f"Formato não suportado: {formato}")
    
    # Converte o timestamp para datetime
    df = converter_timestamp(df, formato)
    
    # Extrai informações do campo file (que é uma string de dicionário Python)
    if 'file' in df.columns:
        # Define o tipo da estrutura do JSON
        file_dtype = pl.Struct([
            pl.Field("filename", pl.Utf8),
            pl.Field("project", pl.Utf8),
            pl.Field("version", pl.Utf8),
            pl.Field("type", pl.Utf8)
        ])
        
        # Para arquivos CSV, precisamos converter as aspas simples para duplas
        if formato == 'csv':
            df = df.with_columns(pl.col("file").str.replace_all("'", '"'))
            
            # Extrai os campos usando json_decode
            df = df.with_columns(
                pl.col("file").str.json_decode(file_dtype).alias("file_info")
            )
        
            # Extrai os campos individuais da estrutura
            df = df.with_columns([
                pl.col("file_info").struct.field("filename").alias("filename"),
                pl.col("file_info").struct.field("project").alias("file_project"),
                pl.col("file_info").struct.field("version").alias("file_version"),
                pl.col("file_info").struct.field("type").alias("file_type")
            ])
        
            # Remove a coluna temporária
            df = df.drop("file_info")
        
        if formato == 'json':
            # Extrai os campos individuais da estrutura
            df = df.with_columns([
                pl.col("file").struct.field("filename").alias("filename"),
                pl.col("file").struct.field("project").alias("file_project"),
                pl.col("file").struct.field("version").alias("file_version"),
                pl.col("file").struct.field("type").alias("file_type")
            ])
    
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
                if df.height > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"Erro ao processar {arquivo}: {e}")
        
        if not dfs:
            print("Nenhum dado válido encontrado nos arquivos")
            return
        
        # Combina todos os DataFrames
        df = pl.concat(dfs)
        
        # Remove linhas com timestamps inválidos
        df = df.drop_nulls(subset=['timestamp'])
        
        # Remove datas muito antigas (antes de 2000) ou futuras
        if 'timestamp' in df.columns:
            
            if formato == 'json':
                df = df.with_columns(pl.from_epoch(pl.col("timestamp")).alias("timestamp"))

            df = df.filter(
                (pl.col("timestamp").dt.year() >= 2000) & 
                (pl.col("timestamp").dt.year() <= datetime.now().year)
            )
        
        # 1. Estatísticas por país
        if 'country_code' in df.columns:
            contagem_paises = df.group_by("country_code").len().sort("len", descending=True)
            print(f"\nEncontrados {contagem_paises.height} países únicos")
            print("Top 5 países por número de downloads:")
            print(contagem_paises.head(5))
        
        # 2. Frequência de downloads ao longo do tempo
        if 'timestamp' in df.columns:
            df = df.with_columns(pl.col("timestamp").dt.date().alias("data"))
            frequencia_downloads = df.group_by("data").len().sort("len", descending=True)
            
            if frequencia_downloads.height > 0:
                print(f"\nFrequência de downloads calculada para {frequencia_downloads.height} datas")
                print("Frequência de downloads por data (top 5):")
                print(frequencia_downloads.head(5))
        
        # 3. Estatísticas de projetos
        if 'project' in df.columns:
            contagem_projetos = df.group_by("project").len().sort("len", descending=True)
            print(f"\nEncontrados {contagem_projetos.height} projetos únicos")
            print("Top 5 projetos por número de downloads:")
            print(contagem_projetos.head(5))
        
        # 4. Estatísticas de arquivos
        if 'filename' in df.columns:
            contagem_arquivos = df.group_by("filename").len().sort("len", descending=True)
            print(f"\nEncontrados {contagem_arquivos.height} arquivos únicos")
            print("Top 5 arquivos por número de downloads:")
            print(contagem_arquivos.head(5))
        
        # 5. Estatísticas de URLs
        if 'url' in df.columns:
            contagem_urls = df.group_by("url").len().sort("len", descending=True)
            print(f"\nEncontrados {contagem_urls.height} URLs únicas")
            print("Top 5 URLs por número de downloads:")
            print(contagem_urls.head(5))
        
        print(f"\nTotal de downloads processados: {df.height}")
        
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
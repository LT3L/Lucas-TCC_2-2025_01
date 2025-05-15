import duckdb
import argparse
import os
from glob import glob
import time
from datetime import datetime

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

def processar_dados_pypi(diretorio_entrada):
    """Processa dados do PyPI do diretório de entrada"""
    try:
        # Detecta o formato e lê os arquivos
        formato = detectar_formato_arquivo(diretorio_entrada)
        if not formato:
            print(f"Nenhum arquivo válido encontrado em {diretorio_entrada}")
            return

        print(f"Formato detectado: {formato}")
        df = ler_arquivo(diretorio_entrada, formato)

        # Criar conexão para processamento SQL
        con = duckdb.connect()

        
        con.register("downloads", df)


        # Processamento SQL (filtros e colunas derivadas)

        if formato == 'csv':
            query = """
            SELECT *,
                JSON_EXTRACT_STRING(REPLACE(file, '''', '"'), '$.filename') as filename,
                JSON_EXTRACT_STRING(REPLACE(file, '''', '"'), '$.project') as file_project,
                JSON_EXTRACT_STRING(REPLACE(file, '''', '"'), '$.version') as file_version,
                JSON_EXTRACT_STRING(REPLACE(file, '''', '"'), '$.type') as file_type
            FROM downloads
            WHERE timestamp IS NOT NULL
            AND timestamp >= '2000-01-01'
            AND timestamp <= CURRENT_TIMESTAMP
            """
        elif formato == 'json':
            query = """
            SELECT to_timestamp(timestamp) as timestamp, 
                country_code, 
                url, 
                project, 
                file,
                JSON_EXTRACT_STRING(file, '$.filename') as filename,
                JSON_EXTRACT_STRING(file, '$.project') as file_project,
                JSON_EXTRACT_STRING(file, '$.version') as file_version,
                JSON_EXTRACT_STRING(file, '$.type') as file_type
            FROM downloads
            WHERE timestamp IS NOT NULL
            AND to_timestamp(timestamp) >= '2000-01-01'
            AND to_timestamp(timestamp) <= CURRENT_TIMESTAMP
            """


        elif formato == 'parquet':
            query = """
            SELECT timestamp as timestamp, 
                country_code, 
                url, 
                project, 
                file,
                JSON_EXTRACT_STRING(file, '$.filename') as filename,
                JSON_EXTRACT_STRING(file, '$.project') as file_project,
                JSON_EXTRACT_STRING(file, '$.version') as file_version,
                JSON_EXTRACT_STRING(file, '$.type') as file_type
            FROM downloads
            WHERE timestamp IS NOT NULL
            AND timestamp >= '2000-01-01'
            AND timestamp <= CURRENT_TIMESTAMP
            """
        
        df = con.execute(query).df()


        con.register("downloads", df)

        # 1. Estatísticas por país
        if 'country_code' in df.columns:
            query = """
            SELECT country_code, COUNT(*) as download_count
            FROM downloads
            GROUP BY country_code
            ORDER BY download_count DESC
            LIMIT 5
            """
            contagem_paises = con.execute(query).df()
            print(f"\nEncontrados {len(contagem_paises)} países únicos")
            print("Top 5 países por número de downloads:")
            print(contagem_paises)

        # 2. Frequência de downloads ao longo do tempo
        if 'timestamp' in df.columns:
            query = """
            SELECT CAST(timestamp AS DATE) as data,
                   COUNT(*) as download_count
            FROM downloads
            GROUP BY data
            ORDER BY download_count DESC
            LIMIT 5
            """
            frequencia_downloads = con.execute(query).df()
            print(f"\nFrequência de downloads calculada para {len(frequencia_downloads)} datas")
            print("Frequência de downloads por data (top 5):")
            print(frequencia_downloads)

        # 3. Estatísticas de projetos
        if 'project' in df.columns:
            query = """
            SELECT project, COUNT(*) as download_count
            FROM downloads
            GROUP BY project
            ORDER BY download_count DESC
            LIMIT 5
            """
            contagem_projetos = con.execute(query).df()
            print(f"\nEncontrados {len(contagem_projetos)} projetos únicos")
            print("Top 5 projetos por número de downloads:")
            print(contagem_projetos)

        # 4. Estatísticas de arquivos
        if 'filename' in df.columns:
            query = """
            SELECT filename, COUNT(*) as download_count
            FROM downloads
            GROUP BY filename
            ORDER BY download_count DESC
            LIMIT 5
            """
            contagem_arquivos = con.execute(query).df()
            print(f"\nEncontrados {len(contagem_arquivos)} arquivos únicos")
            print("Top 5 arquivos por número de downloads:")
            print(contagem_arquivos)

        # 5. Estatísticas de URLs
        if 'url' in df.columns:
            query = """
            SELECT url, COUNT(*) as download_count
            FROM downloads
            GROUP BY url
            ORDER BY download_count DESC
            LIMIT 5
            """
            contagem_urls = con.execute(query).df()
            print(f"\nEncontrados {len(contagem_urls)} URLs únicas")
            print("Top 5 URLs por número de downloads:")
            print(contagem_urls)

        # Total de downloads
        query = "SELECT COUNT(*) as total_downloads FROM downloads"
        total_downloads = con.execute(query).df().iloc[0, 0]
        print(f"\nTotal de downloads processados: {total_downloads}")

    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        import traceback
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
import polars as pl
import argparse
import os
import glob
from pathlib import Path
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

def converter_timestamp(df):
    """Converte timestamps para datetime"""
    try:
        # GitHub commits já têm o timestamp em formato datetime
        return df
    except Exception as e:
        print(f"Erro ao converter timestamps: {e}")
        return df

def ler_arquivos(diretorio, formato):
    """Lê todos os arquivos do diretório de uma vez"""
    try:
        if formato == 'csv':
            # Primeiro, lê um arquivo para inferir o schema
            primeiro_arquivo = sorted(glob(os.path.join(diretorio, '*.csv')))[0]
            schema = pl.read_csv(primeiro_arquivo, try_parse_dates=True, infer_schema_length=10000).schema
            
            # Lê todos os CSVs usando o mesmo schema
            dfs = []
            for f in sorted(glob(os.path.join(diretorio, '*.csv'))):
                try:
                    df = pl.read_csv(f, try_parse_dates=True, schema=schema)
                    dfs.append(df)
                except Exception as e:
                    print(f"Aviso: Erro ao ler arquivo {f}: {e}")
                    continue
            
            if not dfs:
                raise ValueError("Nenhum arquivo CSV válido encontrado")
            
            df = pl.concat(dfs)
            
        elif formato == 'json':
            # Primeiro, lê um arquivo para inferir o schema
            primeiro_arquivo = sorted(glob(os.path.join(diretorio, '*.json')))[0]
            schema = pl.read_ndjson(primeiro_arquivo, infer_schema_length=10000).schema
            
            # Lê todos os JSONs usando o mesmo schema
            dfs = []
            for f in sorted(glob(os.path.join(diretorio, '*.json'))):
                try:
                    df = pl.read_ndjson(f, schema=schema)
                    dfs.append(df)
                except Exception as e:
                    print(f"Aviso: Erro ao ler arquivo {f}: {e}")
                    continue
            
            if not dfs:
                raise ValueError("Nenhum arquivo JSON válido encontrado")
            
            df = pl.concat(dfs)
            
        elif formato == 'parquet':
            # Lê todos os Parquets de uma vez
            df = pl.concat([pl.read_parquet(f) for f in sorted(glob(os.path.join(diretorio, '*.parquet')))])
        else:
            raise ValueError(f"Formato não suportado: {formato}")
        
        # Converte os timestamps para datetime
        df = converter_timestamp(df)
        
        return df
    except Exception as e:
        print(f"Erro ao ler arquivos: {e}")
        print("Detalhes do erro:")
        print(traceback.format_exc())
        return None

def processar_dados_github(df):
    """Processa os dados do GitHub"""
    try:
        # Remove linhas com campos essenciais nulos
        campos_essenciais = ["commit", "author_name", "commit_message", "repository", "commit_time_utc"]
        df = df.drop_nulls(subset=campos_essenciais)
        print(f"📊 Registros após remover nulos essenciais: {df.height}")

        # Converter timestamp Unix para datetime e adicionar colunas derivadas
        df = df.with_columns([
            pl.from_epoch(pl.col("commit_time_utc").cast(pl.Float64), time_unit="s").alias("commit_time_utc"),
            pl.col("commit_message").str.len_chars().alias("message_length")
        ])

        # Filtrar commits inválidos (sem mensagem ou autor)
        df = df.filter(
            (pl.col("commit_message").str.len_chars() > 0) &
            (pl.col("author_name").str.len_chars() > 0)
        )
        print(f"📊 Registros após filtrar commits inválidos: {df.height}")

        return df
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        return None

def main():
    # Argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Script Polars GitHub")
    parser.add_argument('--input', type=str, required=True, help='Caminho para o diretório de entrada')
    args = parser.parse_args()

    # Lista todos os arquivos no diretório
    arquivos = sorted(glob.glob(os.path.join(args.input, "*.*")))
    if not arquivos:
        print(f"❌ Nenhum arquivo encontrado em {args.input}")
        exit(1)

    # Verifica o formato do primeiro arquivo
    primeiro_arquivo = arquivos[0]
    extensao = Path(primeiro_arquivo).suffix.lower()

    print(f"📁 Processando arquivos {extensao} do diretório: {args.input}")
    print(f"📄 Arquivos encontrados: {len(arquivos)}")

    # Lê todos os arquivos de uma vez
    if extensao == '.parquet':
        print("Lendo todos os arquivos parquet...")
        df = pl.concat([pl.read_parquet(f) for f in arquivos])
    elif extensao in ['.json', '.jsonl']:
        print("Lendo todos os arquivos json...")
        df = pl.concat([pl.read_ndjson(f) for f in arquivos])
    elif extensao == '.csv':
        print("Lendo todos os arquivos csv...")
        # Primeiro, lê um arquivo para inferir o schema
        schema = pl.read_csv(arquivos[0], try_parse_dates=True).schema
        # Lê todos os CSVs usando o mesmo schema
        df = pl.concat([pl.read_csv(f, schema=schema) for f in arquivos])
    else:
        print(f"❌ Formato não suportado para o arquivo: {primeiro_arquivo}")
        exit(1)

    print(f"📊 Total de registros lidos: {df.height}")

    # Processa os dados
    df = converter_timestamp(df)
    df = processar_dados_github(df)

    # Verifica se o processamento foi bem sucedido
    if df is None or df.height == 0:
        print("\n❌ Nenhum dado válido após processamento")
        return

    # Criar pasta se não existir
    output_folder = "processed_data/"
    os.makedirs(output_folder, exist_ok=True)

    # Salvar resultados
    output_file = os.path.join(output_folder, "github_commits_processed.parquet")
    df.write_parquet(output_file)
    print(f"✅ Dados processados salvos em: {output_file}")

    # Exibir estatísticas básicas
    print("\n📈 Estatísticas básicas:")
    print(f"Total de commits: {df.height}")
    
    # Calcular estatísticas usando expressões Polars
    stats = df.select([
        pl.col("repository").n_unique().alias("unique_repos"),
        pl.col("author_name").n_unique().alias("unique_authors"),
        pl.col("message_length").mean().alias("avg_message_length"),
        pl.col("commit_time_utc").min().alias("first_commit"),
        pl.col("commit_time_utc").max().alias("last_commit")
    ]).row(0)
    
    unique_repos, unique_authors, avg_length, first_commit, last_commit = stats
    
    print(f"Período: {first_commit} a {last_commit}")
    print(f"Repositórios únicos: {unique_repos}")
    print(f"Autores únicos: {unique_authors}")
    print(f"Tamanho médio das mensagens: {avg_length:.0f} caracteres")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTempo total de processamento: {time.time() - start_time:.2f} segundos")
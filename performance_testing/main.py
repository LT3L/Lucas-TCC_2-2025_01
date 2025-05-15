"""
Script para coletar dados de performance de diferentes bibliotecas de processamento de dados.
Coleta métricas detalhadas sobre características dos datasets e performance de execução.
"""

import time
import subprocess
import pandas as pd
import numpy as np
import sys
import psutil
import threading
import os
import uuid
import platform
import re
import shutil
import datetime
import random
from pathlib import Path
import hashlib
import gc
import signal
import atexit


BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath("app"))
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(BASE_DIR, "datasets"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "datasets_and_models_output"))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
maquina_hash = platform.node().replace(".", "_")
nome_arquivo_saida = f"benchmark_{maquina_hash}_{timestamp}.csv"
caminho_saida_csv = os.path.join(OUTPUT_DIR, nome_arquivo_saida)

# Identificar o modelo do processador
freq = psutil.cpu_freq()
cpu_model = f"{platform.processor()} @ {freq.current:.2f} MHz" if freq else platform.processor()

# Configurações de testes por biblioteca, dataset e formatos
scripts = [
    ("pandas", "performance_testing/pandas/pd_fake_sales.py", "fake_sales", ["csv"], [100, 1_000]),
    ("pandas", "performance_testing/pandas/pd_nyc.py", "nyc_taxi", ["csv", "parquet", "json"], [100, 1000]),
    ("pandas", "performance_testing/pandas/pd_github.py", "github_commits", ["csv", "json", "parquet"], [100, 1000]),
    ("pandas", "performance_testing/pandas/pd_pypi.py", "pypi", ["csv", "json", "parquet"], [100, 1000]),
    
    ("polars", "performance_testing/polars/polars_fake_sales.py", "fake_sales", ["csv"], [100, 1_000]),
    ("polars", "performance_testing/polars/polars_nyc.py", "nyc_taxi", ["csv", "parquet", "json"], [1000]),
    ("polars", "performance_testing/polars/polars_github.py", "github_commits", ["csv", "json", "parquet"], [100, 1000]),
    ("polars", "performance_testing/polars/polars_pypi.py", "pypi", ["csv", "json", "parquet"], [100, 1000]),
    
    ("duckdb", "performance_testing/duckdb/duckdb_fake_sales.py", "fake_sales", ["csv"], [100, 1_000]),
    ("duckdb", "performance_testing/duckdb/duckdb_nyc.py", "nyc_taxi", ["csv", "json", "parquet"], [100, 1000]),
    ("duckdb", "performance_testing/duckdb/duckdb_github.py", "github_commits", ["csv", "json", "parquet"], [100, 1000]),
    ("duckdb", "performance_testing/duckdb/duckdb_pypi.py", "pypi", ["csv", "json", "parquet"], [100, 1000]),
]

random.shuffle(scripts)

MACHINE_INFO = {
    "nucleos_fisicos": psutil.cpu_count(logical=False),
    "nucleos_logicos": psutil.cpu_count(logical=True),
    "frequencia_cpu_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
    "memoria_total_mb": psutil.virtual_memory().total / 1024**2,
}

# Memory monitoring thresholds
MEMORY_THRESHOLD_CRITICAL = 98.5
MEMORY_THRESHOLD_WARNING = 97
MEMORY_CHECK_INTERVAL = 5
SWAP_THRESHOLD = 12

def build_dataset_id(nome: str, formato: str, tamanho_mb: int) -> str:
    return f"{nome}_{formato}_{tamanho_mb}MB" 

def analisar_dataset(caminho: str) -> dict:
    """Analisa características do dataset mantendo todas as colunas importantes."""
    # Determinar o arquivo a ser analisado
    if 'fake_sales' in str(caminho):
        caminho_arquivo = os.path.join(caminho, 'sales.csv')
    else:
        # Para outros datasets, procurar o primeiro arquivo válido na pasta
        if os.path.isdir(caminho):
            arquivos = os.listdir(caminho)
            arquivo_encontrado = False
            for arquivo in arquivos:
                if arquivo.endswith(('.csv', '.parquet', '.json')):
                    caminho_arquivo = os.path.join(caminho, arquivo)
                    arquivo_encontrado = True
                    break
            if not arquivo_encontrado:
                raise FileNotFoundError(f"Nenhum arquivo válido encontrado em {caminho}")
        else:
            caminho_arquivo = caminho

    # print(f"Analisando arquivo: {caminho_arquivo}")
    
    # Verificar se o arquivo existe
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    # Ler o arquivo para contagem total de linhas
    ext = Path(caminho_arquivo).suffix.lower()
    try:
        if ext == '.csv':
            # Para CSV, usar chunksize para contar linhas eficientemente
            total_rows = 0
            for chunk in pd.read_csv(caminho_arquivo, chunksize=100000):
                total_rows += len(chunk)
            # Ler amostra para análise
            df = pd.read_csv(caminho_arquivo, nrows=100000)
        elif ext == '.parquet':
            # Para parquet, ler metadados para contagem total
            total_rows = pd.read_parquet(caminho_arquivo, columns=[]).shape[0]
            # Ler amostra para análise
            df = pd.read_parquet(caminho_arquivo, columns=None)[:100000]
        elif ext == '.json':
            # Para JSON, usar chunksize para contar linhas eficientemente
            total_rows = 0
            for chunk in pd.read_json(caminho_arquivo, lines=True, chunksize=100000):
                total_rows += len(chunk)
            # Ler amostra para análise
            df = pd.read_json(caminho_arquivo, lines=True).head(100000)
        else:
            raise ValueError(f"Extensão não suportada: {ext}")
    except Exception as e:
        raise RuntimeError(f"Erro ao ler o arquivo {caminho_arquivo}: {str(e)}")

    # Análise dos tipos de dados
    total_cols = df.shape[1]
    if total_cols == 0:
        raise ValueError(f"Dataset vazio: {caminho_arquivo}")
    
    # Contagem de tipos específicos
    int_cols = df.select_dtypes(include=['int64']).shape[1]
    float_cols = df.select_dtypes(include=['float64']).shape[1]
    string_cols = df.select_dtypes(include=['object', 'string']).shape[1]
    datetime_cols = df.select_dtypes(include=['datetime']).shape[1]
    numeric_cols = int_cols + float_cols
    other_cols = total_cols - (numeric_cols + string_cols + datetime_cols)

    # Análise de valores únicos e nulos
    unique_ratios = []
    null_ratios = []
    row_sizes = []
    colunas_analisadas = 0
    
    for col in df.columns:
        try:
            # Valores únicos
            unique_ratios.append(df[col].nunique() / len(df))
            # Valores nulos
            null_ratios.append(df[col].isnull().mean())
            # Tamanho da linha
            if df[col].dtype == 'object':
                row_sizes.append(df[col].astype(str).str.len().mean())
            else:
                row_sizes.append(df[col].dtype.itemsize)
            colunas_analisadas += 1
        except (TypeError, ValueError) as e:
            print(f"Aviso: Pulando coluna '{col}' devido a erro: {str(e)}")
            continue

    # Calcular médias apenas para colunas analisadas com sucesso
    avg_unique_ratio = np.mean(unique_ratios) if unique_ratios else 0
    avg_null_ratio = np.mean(null_ratios) if null_ratios else 0
    avg_row_size = sum(row_sizes) if row_sizes else 0

    # Calcular tamanho total
    try:
        if os.path.isdir(caminho):
            total_size = 0
            for root, _, files in os.walk(caminho):
                for file in files:
                    if file.endswith(('.csv', '.parquet', '.json')):
                        total_size += os.path.getsize(os.path.join(root, file))
        else:
            total_size = os.path.getsize(caminho_arquivo)
    except Exception as e:
        raise RuntimeError(f"Erro ao obter tamanho do arquivo/pasta {caminho}: {str(e)}")

    # Criar estatísticas completas
    stats = {
        # Identificadores e tamanhos
        "tamanho_dataset_bytes": total_size,
        "num_linhas": total_rows,  # Número total de linhas
        "num_linhas_amostra": len(df),  # Tamanho da amostra analisada
        "num_colunas": total_cols,
        "num_colunas_analisadas": colunas_analisadas,
        
        # Contagens de tipos
        "num_colunas_numericas": numeric_cols,
        "num_colunas_inteiras": int_cols,
        "num_colunas_float": float_cols,
        "num_colunas_string": string_cols,
        "num_colunas_datetime": datetime_cols,
        "num_colunas_outros": other_cols,
        
        # Percentuais de tipos
        "percentual_numerico": numeric_cols / total_cols if total_cols > 0 else 0,
        "percentual_inteiro": int_cols / total_cols if total_cols > 0 else 0,
        "percentual_float": float_cols / total_cols if total_cols > 0 else 0,
        "percentual_string": string_cols / total_cols if total_cols > 0 else 0,
        "percentual_datetime": datetime_cols / total_cols if total_cols > 0 else 0,
        "percentual_outros": other_cols / total_cols if total_cols > 0 else 0,
        
        # Métricas de qualidade (calculadas na amostra)
        "media_valores_unicos": avg_unique_ratio,
        "media_valores_nulos": avg_null_ratio,
        "tamanho_medio_linha": avg_row_size
    }
    
    # print("\nEstatísticas do dataset coletadas:")
    # print(f"Total de linhas: {total_rows:,}")
    # print(f"Tamanho da amostra analisada: {len(df):,}")
    # for key, value in stats.items():
    #     if key not in ["num_linhas", "num_linhas_amostra"]:  # Já mostrados acima
    #         print(f"{key}: {value}")
    
    return stats

def registrar_execucao_benchmark(
    biblioteca, dataset_path, dataset_nome, dataset_formato,
    tamanho_nominal_mb, tempo_execucao, cpu_medio, memoria_media,
    leitura_bytes, escrita_bytes, tem_joins=False, tem_groupby=False,
    status="completed", termination_reason=None
):
    id_execucao = str(uuid.uuid4())
    dataset_id = build_dataset_id(dataset_nome, dataset_formato, tamanho_nominal_mb)
    
    # Obter estatísticas do dataset
    dataset_stats = analisar_dataset(dataset_path)
    
    # Criar registro com todas as informações
    registro = {
        # Identificadores
        "dataset_id": dataset_id,
        "id_execucao": id_execucao,
        
        # Informações da biblioteca e dataset
        "biblioteca": biblioteca,
        "dataset_nome": dataset_nome,
        "dataset_formato": dataset_formato,
        "tamanho_dataset_nominal_mb": tamanho_nominal_mb,
        
        # Métricas de performance
        "tempo_execucao": tempo_execucao,
        "cpu_medio_execucao": cpu_medio,
        "memoria_media_execucao": memoria_media,
        "leitura_bytes": leitura_bytes,
        "escrita_bytes": escrita_bytes,
        
        # Flags de operação
        "tem_joins": tem_joins,
        "tem_groupby": tem_groupby,
        
        # Status da execução
        "status": status,
        "termination_reason": termination_reason,
        
        # Informações da máquina
        **MACHINE_INFO,
        
        # Estatísticas do dataset
        **dataset_stats
    }

    # # Debug: Verificar colunas antes de salvar
    # print("\nColunas que serão salvas no CSV:")
    # for key in registro.keys():
    #     print(f"- {key}")

    arquivo = caminho_saida_csv
    with open(arquivo, "a") as f:
        df = pd.DataFrame([registro])
        df.to_csv(f, header=not os.path.exists(arquivo) or os.stat(arquivo).st_size == 0, index=False)
        
        # # Debug: Verificar se o arquivo foi criado/atualizado
        # if os.path.exists(arquivo):
            # print(f"\nArquivo CSV atualizado: {arquivo}")
            # Ler as últimas linhas para verificar
            # try:
                # df_check = pd.read_csv(arquivo)
                # print(f"Colunas no arquivo CSV: {df_check.columns.tolist()}")
            # except Exception as e:
            #    print(f"Erro ao verificar arquivo CSV: {str(e)}")

# Número de execuções
num_execucoes = 10

def extrair_tamanho_nominal(path):
    match = re.search(r'_(\d+)(mb|gb)', path.lower())
    if match:
        valor = int(match.group(1))
        unidade = match.group(2)
        return valor * 1024 if unidade == 'gb' else valor
    return None

def monitorar_recursos(uso_detalhado, flag_parar, nome, execucao, pid):
    process = psutil.Process(pid)
    last_memory_check = time.time()
    consecutive_critical_checks = 0
    termination_reason = None
    
    try:
        critical_checks_count = 0
        while not flag_parar.is_set():
            try:
                # Check memory usage periodically
                current_time = time.time()
                if current_time - last_memory_check >= MEMORY_CHECK_INTERVAL:
                    memory = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    
                    # Increment counter if either memory or swap is critical
                    if memory.percent >= MEMORY_THRESHOLD_CRITICAL or swap.percent >= SWAP_THRESHOLD:
                        critical_checks_count += 1
                        print(f"⚠️ Critical check #{critical_checks_count}: Memory at {memory.percent:.1f}%, Swap at {swap.percent:.1f}%")
                        
                        if critical_checks_count >= 20:  # Terminate after accumulating 10 critical checks
                            print("🛑 Accumulated critical resource usage - initiating process termination")
                            termination_reason = "resource_limit"
                            try:
                                process.terminate()
                                process.wait(timeout=5)
                            except:
                                try:
                                    process.kill()
                                except:
                                    pass
                            flag_parar.set()
                            return termination_reason
                            
                    last_memory_check = current_time

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                cpu_usage = process.cpu_percent(interval=0.2)
                mem_usage = process.memory_info().rss / 1024 ** 2
                total_mem = psutil.virtual_memory().total / (1024 * 1024)
                swap_usage = psutil.swap_memory().percent
                power_usage = None
                open_files = len(process.open_files())
                threads = process.num_threads()
                io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
                read_bytes = io_counters.read_bytes if io_counters else 0
                write_bytes = io_counters.write_bytes if io_counters else 0

                uso_detalhado.append({
                    "Timestamp": timestamp,
                    "Biblioteca": nome,
                    "Execução": execucao,
                    "CPU (%)": cpu_usage,
                    "Memória Processo (MB)": mem_usage,
                    "Memória Total (MB)": total_mem,
                    "Swap (%)": swap_usage,
                    "Consumo Energia (W)": power_usage,
                    "Arquivos Abertos": open_files,
                    "Threads": threads,
                    "Leitura Bytes": read_bytes,
                    "Escrita Bytes": write_bytes,
                })

                # Check if process is still running
                if not process.is_running():
                    break

                time.sleep(0.2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                termination_reason = "process_not_found"
                break
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                termination_reason = f"monitoring_error: {str(e)}"
                break
    finally:
        try:
            process.close()
        except:
            pass
        return termination_reason

def wait_for_memory():
    """Wait until system memory is available"""
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        memory = psutil.virtual_memory()
        if memory.percent < 85:  # If memory usage is below 85%
            return True
            
        print(f"⚠️ High memory usage detected ({memory.percent:.1f}%). Waiting for resources to free up...")
        gc.collect()  # Force garbage collection
        time.sleep(10)  # Wait 10 seconds before checking again
        retry_count += 1
    
    print("⚠️ Memory usage still high after maximum retries. Proceeding with caution...")
    return False

def cleanup_resources():
    """Cleanup function to be called on exit"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clean up processed data directory
        pasta = "processed_data"
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.makedirs(pasta, exist_ok=True)
        
    except Exception as e:
        print(f"Error during cleanup: {e}")



def medir_tempo(script, nome, dataset_nome, dataset_formato, dataset_path, tamanho):
    tempos = []
    for i in range(num_execucoes):
        if not wait_for_memory():
            print(f"Skipping execution {i+1} due to high memory usage")
            registrar_execucao_benchmark(
                biblioteca=nome,
                dataset_path=dataset_path,
                dataset_nome=dataset_nome,
                dataset_formato=dataset_formato,
                tempo_execucao=0,
                tamanho_nominal_mb=tamanho,
                cpu_medio=0,
                memoria_media=0,
                leitura_bytes=0,
                escrita_bytes=0,
                status="dnf",
                termination_reason="high_initial_memory"
            )
            continue
            
        inicio = time.perf_counter()
        print(f"Starting execution: {nome} - {dataset_nome} ({tamanho}MB {dataset_formato})")
        
        try:
            process = subprocess.Popen([sys.executable, script, "--input", dataset_path])
            pid = process.pid
            flag_parar = threading.Event()
            uso_detalhado = []
            
            monitor_thread = threading.Thread(
                target=monitorar_recursos,
                args=(uso_detalhado, flag_parar, nome, i + 1, pid),
                daemon=True
            )
            monitor_thread.start()

            process.wait()
            flag_parar.set()
            termination_reason = monitor_thread.join(timeout=5)

            status = "completed"
            if process.returncode != 0:
                print(f"⚠️ Process terminated with non-zero exit code: {process.returncode}")
                status = "error"
                termination_reason = f"exit_code_{process.returncode}"

            fim = time.perf_counter()
            tempo_execucao = fim - inicio

            cpu_vals = [x["CPU (%)"] for x in uso_detalhado if "CPU (%)" in x]
            mem_vals = [x["Memória Processo (MB)"] for x in uso_detalhado if "Memória Processo (MB)" in x]
            swap_vals = [x["Swap (%)"] for x in uso_detalhado if "Swap (%)" in x]

            media_cpu = np.mean(cpu_vals) if cpu_vals else 0.0
            media_memoria = np.mean(mem_vals) if mem_vals else 0.0
            media_swap = np.mean(swap_vals) if swap_vals else 0.0
            leitura_bytes = uso_detalhado[-1]["Leitura Bytes"] if uso_detalhado else 0
            escrita_bytes = uso_detalhado[-1]["Escrita Bytes"] if uso_detalhado else 0

            print(f"Execution completed: CPU {media_cpu:.1f}%, Memory {media_memoria:.1f}MB, Swap {media_swap:.1f}%")

            registrar_execucao_benchmark(
                biblioteca=nome,
                dataset_path=dataset_path,
                dataset_nome=dataset_nome,
                dataset_formato=dataset_formato,
                tempo_execucao=tempo_execucao,
                tamanho_nominal_mb=tamanho,
                cpu_medio=media_cpu,
                memoria_media=media_memoria,
                leitura_bytes=leitura_bytes,
                escrita_bytes=escrita_bytes,
                status=status,
                termination_reason=termination_reason
            )

        except Exception as e:
            print(f"Error during execution: {e}")
            registrar_execucao_benchmark(
                biblioteca=nome,
                dataset_path=dataset_path,
                dataset_nome=dataset_nome,
                dataset_formato=dataset_formato,
                tempo_execucao=0,
                tamanho_nominal_mb=tamanho,
                cpu_medio=0,
                memoria_media=0,
                leitura_bytes=0,
                escrita_bytes=0,
                status="error",
                termination_reason=f"exception: {str(e)}"
            )
        finally:
            # Clean up after each execution
            gc.collect()
            time.sleep(2)  # Small delay between executions

def main():
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        cleanup_resources()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Executar os testes para cada tamanho
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    try:
        for nome, script, dataset_nome, dataset_formato, tamanhos in scripts:
            if isinstance(dataset_formato, list):
                formatos = dataset_formato
            else:
                formatos = [dataset_formato]
            for formato in formatos:
                for tamanho in tamanhos:
                    dataset_path = os.path.join(DATASET_DIR, dataset_nome, formato, f"{tamanho}MB")

                    print("Running", script, nome, dataset_nome, formato, dataset_path)
                    medir_tempo(script, nome, dataset_nome, formato, dataset_path, tamanho)

        print("Testes concluídos e dados registrados em", caminho_saida_csv)
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()

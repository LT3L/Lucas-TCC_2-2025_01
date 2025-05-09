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
    # ("pandas", "performance_testing/pd_fake_sales.py", "fake_sales", ["csv"], [100, 1_000]),
    # ("polars", "performance_testing/polars_fake_sales.py", "fake_sales", ["csv"], [100, 1_000]),
    # ("duckdb", "performance_testing/duckdb_fake_sales.py", "fake_sales", ["csv"], [100, 1_000]),
    
    # ("pyspark", "performance_testing/pyspark_fake_sales.py", "fake_sales", ["csv"], [100, 1000]),


    # ("pandas", "performance_testing/pd_nyc.py", "nyc_taxi", ["csv", "parquet", "json"], [100, 1000]),
    ("pandas", "performance_testing/pd_github.py", "github_commits", ["csv", "json", "parquet"], [100, 1000]), # , 10000, 50000]),
    ("pandas", "performance_testing/pd_pypi.py", "pypi", ["csv", "json", "parquet"], [100, 1000]), # , 10000, 50000]),
    
    #
    # ("polars", "performance_testing/polars_github.py", "github_commits", ["csv"], [100, 1000]), # , 10000, 50000]),
    # ### , "json" Não foi usado com polars por conta de não lidar com os aninhamentos
    # ("polars", "performance_testing/polars_nyc.py", "nyc_taxi", ["csv", "json", "parquet"], [100, 1000, ]),
    #
    # ("duckdb", "performance_testing/duckdb_github.py", "github_commits", ["csv", "json"], [100, 1000]), # , 10000, 50000]),
    # ("duckdb", "performance_testing/duckdb_nyc.py", "nyc_taxi", ["csv", "parquet", "json"], [100, 1000,  ]),
    #
    # # ("pyspark", "performance_testing/pyspark_github.py", "github_commits", ["csv", "json"], [100, 1000, 10000]), # , 10000, 50000]),
    # # ("pyspark", "performance_testing/pyspark_nyc.py", "nyc_taxi", "csv", [100, 1000, ]),
    # # ("pyspark", "performance_testing/pyspark_nyc.py", "nyc_taxi", "parquet", [100, 1000, ]),
    # # ("pyspark", "performance_testing/pyspark_nyc.py", "nyc_taxi", "json", [100, 1000,  ]),
]

random.shuffle(scripts)

MACHINE_INFO = {     # calculado uma vez
    "cpu_model": platform.processor(),
    "nucleos_fisicos": psutil.cpu_count(logical=False),
    "nucleos_logicos": psutil.cpu_count(logical=True),
    "frequencia_cpu_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
    "memoria_total_mb": psutil.virtual_memory().total / 1024**2,
    "disco_total_gb": psutil.disk_usage('/').total / 1024**3,
    "sistema_operacional": platform.system(),
}

DATASET_CACHE: dict[str, dict] = {}   # ➜ evita re‐análise

# utilidades ----------------------------------------------------------------
def build_dataset_id(nome: str, formato: str, tamanho_mb: int) -> str:
    return f"{nome}_{formato}_{tamanho_mb}MB"

def analisar_dataset(caminho: str) -> dict:
    """Lê no máximo 5 000 linhas p/ extrair estatísticas leves."""
    if caminho in DATASET_CACHE:
        return DATASET_CACHE[caminho]

    ext = Path(caminho).suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(caminho, nrows=5_000)
        elif ext == '.parquet':
            df = pd.read_parquet(caminho, columns=None)[:5_000]
        elif ext == '.json':
            df = pd.read_json(caminho, lines=True).head(5_000)
        else:
            DATASET_CACHE[caminho] = {}
            return {}
    except Exception:
        DATASET_CACHE[caminho] = {}
        return {}

    # tentativa de converter objects → datetime
    for col in df.select_dtypes('object'):
        conv = pd.to_datetime(df[col], errors='coerce', utc=True)
        if conv.notna().mean() > 0.8:
            df[col] = conv

    stats = {
        "tamanho_dataset_bytes": Path(caminho).stat().st_size,
        "num_linhas": len(df),
        "num_colunas": df.shape[1],
        "percentual_numerico": df.select_dtypes('number').shape[1] / df.shape[1],
        "percentual_string":  df.select_dtypes(['object', 'string']).shape[1] / df.shape[1],
        "percentual_datetime": df.select_dtypes('datetime').shape[1] / df.shape[1],
    }
    DATASET_CACHE[caminho] = stats
    return stats

# Número de execuções
num_execucoes = 1

def extrair_tamanho_nominal(path):
    match = re.search(r'_(\d+)(mb|gb)', path.lower())
    if match:
        valor = int(match.group(1))
        unidade = match.group(2)
        return valor * 1024 if unidade == 'gb' else valor
    return None

def registrar_execucao_benchmark(
    biblioteca, dataset_path, dataset_nome, dataset_formato,
    tamanho_nominal_mb,  # ◄- NOVO
    tempo_execucao, cpu_medio, memoria_media, leitura_bytes, escrita_bytes,
    tem_joins=False, tem_groupby=False
):
    id_execucao = str(uuid.uuid4())

    dataset_id = build_dataset_id(dataset_nome, dataset_formato, tamanho_nominal_mb)


    registro = {
        "dataset_id": dataset_id,  # NOVA COLUNA
        "id_execucao": id_execucao,
        "biblioteca": biblioteca,
        "dataset_nome": dataset_nome,
        "dataset_formato": dataset_formato,
        "tamanho_dataset_nominal_mb": tamanho_nominal_mb,
        "tempo_execucao": tempo_execucao,
        "cpu_medio_execucao": cpu_medio,
        "memoria_media_execucao": memoria_media,
        "leitura_bytes": leitura_bytes,
        "escrita_bytes": escrita_bytes,
        "tem_joins": tem_joins,
        "tem_groupby": tem_groupby,
        ** MACHINE_INFO,
        **analisar_dataset(dataset_path)
    }

    arquivo = caminho_saida_csv

    with open(arquivo, "a") as f:
        pd.DataFrame([registro]).to_csv(f, header=not os.path.exists(arquivo) or os.stat(arquivo).st_size == 0, index=False)

def monitorar_recursos(uso_detalhado, flag_parar, nome, execucao, pid):
    process = psutil.Process(pid)
    while not flag_parar.is_set():
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            cpu_usage = process.cpu_percent(interval=0.2)
            mem_usage = process.memory_info().rss / 1024 ** 2
            total_mem = psutil.virtual_memory().total / (1024 * 1024)
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
                "Consumo Energia (W)": power_usage,
                "Arquivos Abertos": open_files,
                "Threads": threads,
                "Leitura Bytes": read_bytes,
                "Escrita Bytes": write_bytes,
            })

            time.sleep(0.2)
        except psutil.NoSuchProcess:
            break

def medir_tempo(script, nome, dataset_nome, dataset_formato, dataset_path, tamanho):
    tempos = []
    for i in range(num_execucoes):
        inicio = time.perf_counter()
        print("dataset_path:", dataset_path)
        process = subprocess.Popen([sys.executable, script, "--input", dataset_path])
        pid = process.pid
        flag_parar = threading.Event()
        uso_detalhado = []
        monitor_thread = threading.Thread(target=monitorar_recursos, args=(uso_detalhado, flag_parar, nome, i + 1, pid))
        monitor_thread.start()

        process.wait()
        flag_parar.set()
        monitor_thread.join()

        fim = time.perf_counter()
        tempo_execucao = fim - inicio

        cpu_vals = [x["CPU (%)"] for x in uso_detalhado if "CPU (%)" in x]
        mem_vals = [x["Memória Processo (MB)"] for x in uso_detalhado if "Memória Processo (MB)" in x]

        media_cpu = np.mean(cpu_vals) if cpu_vals else 0.0
        media_memoria = np.mean(mem_vals) if mem_vals else 0.0
        leitura_bytes = uso_detalhado[-1]["Leitura Bytes"] if uso_detalhado else 0
        escrita_bytes = uso_detalhado[-1]["Escrita Bytes"] if uso_detalhado else 0

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
            tem_joins=False,
            tem_groupby=False
        )

    pasta = "processed_data"

    # Remove a pasta inteira (com tudo dentro)
    if os.path.exists(pasta):
        shutil.rmtree(pasta)

    # Recria a pasta vazia (opcional)
    os.makedirs(pasta, exist_ok=True)


def main():
    # Executar os testes para cada tamanho

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

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


if __name__ == "__main__":
    main()

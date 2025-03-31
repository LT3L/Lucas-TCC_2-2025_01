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

# Identificar o modelo do processador
freq = psutil.cpu_freq()
cpu_model = f"{platform.processor()} @ {freq.current:.2f} MHz" if freq else platform.processor()

# Configurações de testes por biblioteca, dataset e formatos
scripts = [
    # ("pandas", "pd_nyc.py", "nyc_taxi", "csv", [100, 1000, ]), # 10000, 50000]),
    # ("pandas", "pd_nyc.py", "nyc_taxi", "parquet", [100, 1000]),
    # ("pandas", "pd_nyc.py", "nyc_taxi", "json", [100, 1000,]),
    # ("polars", "polars_nyc.py", "nyc_taxi", "csv", [100, 1000, ]),
    # ("polars", "polars_nyc.py", "nyc_taxi", "parquet", [100, 1000, ]),
    # ("polars", "polars_nyc.py", "nyc_taxi", "json", [100, 1000,  ]),

    ("pyspark", "pyspark_nyc.py", "nyc_taxi", "csv", [100, 1000, ]),
    ("pyspark", "pyspark_nyc.py", "nyc_taxi", "parquet", [100, 1000, ]),
    ("pyspark", "pyspark_nyc.py", "nyc_taxi", "json", [100, 1000,  ]),
]

# Número de execuções
num_execucoes = 1

def extrair_tamanho_nominal(path):
    match = re.search(r'_(\d+)(mb|gb)', path.lower())
    if match:
        valor = int(match.group(1))
        unidade = match.group(2)
        return valor * 1024 if unidade == 'gb' else valor
    return None

def coletar_caracteristicas_maquina():
    return {
        "cpu_model": platform.processor(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "frequencia_cpu_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "memoria_total_mb": psutil.virtual_memory().total / (1024 ** 2),
        "disco_total_gb": psutil.disk_usage('/').total / (1024 ** 3),
        "sistema_operacional": platform.system(),
    }

def analisar_dataset(caminho):
    ext = os.path.splitext(caminho)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(caminho, nrows=5000)
        elif ext == '.parquet':
            df = pd.read_parquet(caminho)
        elif ext == '.json':
            df = pd.read_json(caminho, lines=True, nrows=5000)
        else:
            return {}
    except Exception:
        return {}

    tipos = df.dtypes
    total = len(tipos)
    return {
        "tamanho_dataset_bytes": os.path.getsize(caminho),
        "num_linhas": len(df),
        "num_colunas": df.shape[1],
        "percentual_numerico": tipos.isin(['int64', 'float64']).mean(),
        "percentual_string": sum(tipos == 'object') / total,
        "percentual_datetime": sum(tipos == 'datetime64[ns]') / total,
    }

def registrar_execucao_benchmark(
    biblioteca, dataset_path, dataset_nome, dataset_formato,
    tempo_execucao, cpu_medio, memoria_media, leitura_bytes, escrita_bytes,
    tem_joins=False, tem_groupby=False
):
    id_execucao = str(uuid.uuid4())
    maquina = coletar_caracteristicas_maquina()
    dados = analisar_dataset(dataset_path)
    tamanho_nominal = extrair_tamanho_nominal(dataset_path)

    registro = {
        "id_execucao": id_execucao,
        "biblioteca": biblioteca,
        "dataset_nome": dataset_nome,
        "dataset_formato": dataset_formato,
        "tamanho_dataset_nominal_mb": tamanho_nominal,
        "tempo_execucao": tempo_execucao,
        "cpu_medio_execucao": cpu_medio,
        "memoria_media_execucao": memoria_media,
        "leitura_bytes": leitura_bytes,
        "escrita_bytes": escrita_bytes,
        "tem_joins": tem_joins,
        "tem_groupby": tem_groupby,
        **maquina,
        **dados
    }

    arquivo = "../datasets_and_models_output/dataset_benchmark.csv"
    if os.path.exists(arquivo):
        df = pd.read_csv(arquivo)
        df = pd.concat([df, pd.DataFrame([registro])], ignore_index=True)
    else:
        df = pd.DataFrame([registro])

    df.to_csv(arquivo, index=False)

def monitorar_recursos(uso_detalhado, flag_parar, nome, execucao, pid):
    process = psutil.Process(pid)
    while not flag_parar.is_set():
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")
            cpu_usage = psutil.cpu_percent(interval=1)
            mem_usage = process.memory_info().rss / (1024 * 1024)
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
        except psutil.NoSuchProcess:
            break

def medir_tempo(script, nome, dataset_nome, dataset_formato, dataset_path):
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

        media_cpu = np.mean([x["CPU (%)"] for x in uso_detalhado if "CPU (%)" in x])
        media_memoria = np.mean([x["Memória Processo (MB)"] for x in uso_detalhado if "Memória Processo (MB)" in x])
        leitura_bytes = uso_detalhado[-1]["Leitura Bytes"] if uso_detalhado else 0
        escrita_bytes = uso_detalhado[-1]["Escrita Bytes"] if uso_detalhado else 0

        registrar_execucao_benchmark(
            biblioteca=nome,
            dataset_path=dataset_path,
            dataset_nome=dataset_nome,
            dataset_formato=dataset_formato,
            tempo_execucao=tempo_execucao,
            cpu_medio=media_cpu,
            memoria_media=media_memoria,
            leitura_bytes=leitura_bytes,
            escrita_bytes=escrita_bytes,
            tem_joins=False,
            tem_groupby=False
        )


def main():
    # Executar os testes para cada tamanho
    for nome, script, dataset_nome, dataset_formato, tamanhos in scripts:
        for tamanho in tamanhos:

            dataset_path = f"/Users/lucas.lima/Documents/Projects/TCC_2/datasets/{dataset_nome}/{dataset_formato}/amostra_{tamanho}MB.{dataset_formato}"


            print("Running", script, nome, dataset_nome, dataset_formato, dataset_path)
            medir_tempo(script, nome, dataset_nome, dataset_formato, dataset_path)

    print("Testes concluídos e dados registrados em dataset_benchmark.csv")


if __name__ == "__main__":
    main()

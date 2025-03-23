import time
import subprocess
import pandas as pd
import numpy as np
import sys
import psutil
import threading
from datetime import datetime  # ✅ Adicionado para timestamps com microsegundos

# Identificar o modelo do processador
def get_cpu_model():
    try:
        result = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        return result.decode().strip()
    except Exception:
        return "Unknown"

cpu_model = get_cpu_model()

scripts = [
    ("pandas", "pd_nyc.py"),
    ("polars", "polars_nyc.py"),
]

num_execucoes = 1
resultados_detalhados = []
resultados_finais = []
uso_global = []

def monitorar_todos_processos(lista_uso_global, flag_parar):
    while not flag_parar.is_set():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # ✅ Correto agora
        for proc in psutil.process_iter(attrs=["pid", "ppid", "name", "cmdline", "memory_info"]):
            try:
                info = proc.info
                lista_uso_global.append({
                    "Timestamp": timestamp,
                    "PID": info["pid"],
                    "PPID": info["ppid"],
                    "Nome": info["name"],
                    "Cmdline": " ".join(info["cmdline"]) if info["cmdline"] else "",
                    "CPU (%)": proc.cpu_percent(interval=None),
                    "Memória (MB)": info["memory_info"].rss / (1024 * 1024)
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.01)

def medir_tempo(script, nome, caminho_arquivo):
    tempos = []
    for i in range(num_execucoes):
        print(f"🚀 Executando {nome} com {caminho_arquivo} (Execução {i+1})")
        inicio = time.perf_counter()
        timestamp_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # ✅ Correto

        process = subprocess.Popen([sys.executable, script, "--input", caminho_arquivo])
        pid = process.pid

        process.wait()

        fim = time.perf_counter()
        timestamp_fim = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # ✅ Correto
        tempo_execucao = fim - inicio
        tempos.append(tempo_execucao)

        resultados_detalhados.append({
            "Biblioteca": nome,
            "Arquivo": caminho_arquivo,
            "Execução": i + 1,
            "Tempo (s)": tempo_execucao,
            "Timestamp Início": timestamp_inicio,
            "Timestamp Fim": timestamp_fim,
            "PID": pid
        })

    return np.mean(tempos), np.std(tempos)

flag_monitor_global = threading.Event()
monitor_global_thread = threading.Thread(target=monitorar_todos_processos, args=(uso_global, flag_monitor_global))
monitor_global_thread.start()

arquivos_para_testar = [
    "NYC_sized/parquet/amostra_10MB.parquet",
    "NYC_sized/parquet/amostra_100MB.parquet",
    "NYC_sized/parquet/amostra_1000MB.parquet",
    "NYC_sized/csv/amostra_10MB.csv",
    "NYC_sized/csv/amostra_100MB.csv",
    "NYC_sized/csv/amostra_1000MB.csv",
    "NYC_sized/json/amostra_10MB.json",
    "NYC_sized/json/amostra_100MB.json",
    "NYC_sized/json/amostra_1000MB.json",
]

for caminho_arquivo in arquivos_para_testar:
    for nome, script in scripts:
        media, desvio = medir_tempo(script, nome, caminho_arquivo)
        resultados_finais.append({
            "Biblioteca": nome,
            "Arquivo": caminho_arquivo,
            "Média (s)": media,
            "Desvio Padrão (s)": desvio,
            "Modelo CPU": cpu_model
        })

time.sleep(1)
flag_monitor_global.set()
monitor_global_thread.join()

pd.DataFrame(resultados_detalhados).to_csv("resultados_detalhados.csv", index=False)
pd.DataFrame(resultados_finais).to_csv("resultados_finais.csv", index=False)
pd.DataFrame(uso_global).to_csv("uso_global.csv", index=False)

print(f"✅ Testes concluídos. CPU: {cpu_model}")
print("📄 Resultados salvos em:")
print(" - resultados_detalhados.csv")
print(" - resultados_finais.csv")
print(" - uso_global.csv")
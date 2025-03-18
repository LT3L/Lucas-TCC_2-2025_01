import time
import subprocess
import pandas as pd
import numpy as np
import sys
import psutil
import threading

# Lista de scripts para testar
scripts = [
    ("pandas", "pd_nyc.py"),
    ("polars", "polars_nyc.py"),
]

# Número de execuções
num_execucoes = 10

def monitorar_recursos(uso_cpu, uso_memoria, flag_parar, nome, execucao, resultados_granulares):
    """Monitora CPU e Memória a cada segundo durante a execução."""
    while not flag_parar.is_set():
        cpu = psutil.cpu_percent(interval=1)
        memoria = psutil.virtual_memory().percent
        uso_cpu.append(cpu)
        uso_memoria.append(memoria)
        resultados_granulares.append({
            "Biblioteca": nome,
            "Execução": execucao,
            "Timestamp": time.time(),
            "CPU (%)": cpu,
            "Memória (%)": memoria
        })

# Lista para armazenar os resultados
resultados_detalhados = []
resultados_finais = []
resultados_granulares = []

# Função para executar um script e medir tempo, CPU e memória
def medir_tempo(script, nome):
    tempos = []
    uso_cpu_exec = []
    uso_memoria_exec = []

    for i in range(num_execucoes):  # Executa várias vezes
        inicio = time.perf_counter()
        uso_cpu = []
        uso_memoria = []
        flag_parar = threading.Event()
        monitor_thread = threading.Thread(target=monitorar_recursos, args=(uso_cpu, uso_memoria, flag_parar, nome, i + 1, resultados_granulares))
        monitor_thread.start()

        subprocess.run([sys.executable, script], check=True)  # Usa o mesmo Python da execução

        flag_parar.set()
        monitor_thread.join()
        fim = time.perf_counter()

        tempo_execucao = fim - inicio
        tempos.append(tempo_execucao)
        uso_cpu_exec.append(np.mean(uso_cpu))
        uso_memoria_exec.append(np.mean(uso_memoria))

        resultados_detalhados.append({
            "Biblioteca": nome,
            "Execução": i + 1,
            "Tempo (s)": tempo_execucao,
            "CPU Média (%)": np.mean(uso_cpu),
            "Memória Média (%)": np.mean(uso_memoria)
        })

    return np.mean(tempos), np.std(tempos), np.mean(uso_cpu_exec), np.mean(uso_memoria_exec)


# Executar os testes e armazenar resultados
for nome, script in scripts:
    media, desvio, cpu_media, memoria_media = medir_tempo(script, nome)
    resultados_finais.append({
        "Biblioteca": nome,
        "Média (s)": media,
        "Desvio Padrão (s)": desvio,
        "CPU Média (%)": cpu_media,
        "Memória Média (%)": memoria_media
    })

# Criar DataFrames com os resultados
df_detalhado = pd.DataFrame(resultados_detalhados)
df_finais = pd.DataFrame(resultados_finais)
df_granulares = pd.DataFrame(resultados_granulares)

# Salvar resultados em CSV
df_detalhado.to_csv("resultados_detalhados.csv", index=False)
df_finais.to_csv("resultados_finais.csv", index=False)
df_granulares.to_csv("resultados_granulares.csv", index=False)

# Mensagem de conclusão
print("Testes concluídos. Resultados salvos em 'resultados_detalhados.csv', 'resultados_finais.csv' e 'resultados_granulares.csv'.")

import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV com os dados granulares
file_path = "resultados_granulares.csv"
df_granulares = pd.read_csv(file_path)

# Converter timestamps para iniciar no tempo 0 por execução
df_granulares["Timestamp"] = df_granulares.groupby(["Biblioteca", "Execução"])["Timestamp"].transform(lambda x: x - x.min())

# Criar o gráfico
plt.figure(figsize=(10, 6))

# Plotar a média da CPU e memória para cada ferramenta ao longo do tempo
for biblioteca in df_granulares["Biblioteca"].unique():
    df_bib = df_granulares[df_granulares["Biblioteca"] == biblioteca]
    df_media = df_bib.groupby("Timestamp")[["CPU (%)", "Memória (%)"]].mean()
    plt.plot(df_media.index, df_media["CPU (%)"], label=f"CPU {biblioteca}")
    plt.plot(df_media.index, df_media["Memória (%)"], linestyle="dashed", label=f"Memória {biblioteca}")

# Configurar rótulos e título
plt.xlabel("Tempo de Execução (s)")
plt.ylabel("Uso (%)")
plt.title("Uso Médio de CPU e Memória Durante a Execução")
plt.legend()
plt.grid()

# Exibir o gráfico
plt.show()

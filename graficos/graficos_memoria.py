# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Recarrega e filtra dados
# uso_global = pd.read_csv("uso_global.csv")
# uso_global = uso_global[uso_global['Cmdline'].str.contains('.py', na=False)].copy()
#
# # Extrai informações
# def extrair_biblioteca(cmd):
#     if 'pd_nyc.py' in cmd:
#         return 'Pandas'
#     elif 'polars_nyc.py' in cmd:
#         return 'Polars'
#     return 'Outro'
#
# def extrair_tamanho(cmd):
#     for tamanho in ['10MB', '100MB', '1GB', '1000MB']:
#         if tamanho in cmd:
#             return tamanho
#     return 'Desconhecido'
#
# uso_global['Biblioteca'] = uso_global['Cmdline'].apply(extrair_biblioteca)
# uso_global['Arquivo'] = uso_global['Cmdline'].apply(extrair_tamanho)
# uso_global['Memória (MB)'] = pd.to_numeric(uso_global['Memória (MB)'], errors='coerce')
#
# # Filtra apenas Pandas e Polars
# uso_global = uso_global[uso_global['Biblioteca'].isin(['Pandas', 'Polars'])]
#
# # Gera um gráfico separado para cada tamanho de arquivo
# for tamanho in uso_global['Arquivo'].unique():
#     df_subset = uso_global[uso_global['Arquivo'] == tamanho]
#     if df_subset.empty:
#         continue
#
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=df_subset, x='Biblioteca', y='Memória (MB)', palette='Set2')
#     plt.title(f'Uso de Memória por Biblioteca - Arquivo {tamanho}')
#     ymax = df_subset['Memória (MB)'].max() * 1.05
#     plt.ylim(0, ymax)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Carrega o CSV corrigido
uso_global = pd.read_csv("uso_global.csv")

# Filtro e limpeza
uso_global = uso_global[uso_global['Cmdline'].str.contains('.py', na=False)].copy()
uso_global['Timestamp'] = pd.to_datetime(uso_global['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
uso_global['Memória (MB)'] = pd.to_numeric(uso_global['Memória (MB)'], errors='coerce')
uso_global['CPU (%)'] = pd.to_numeric(uso_global['CPU (%)'], errors='coerce')

# Extrai biblioteca e tamanho de arquivo
def extrair_biblioteca(cmd):
    if 'pd_nyc.py' in cmd:
        return 'Pandas'
    elif 'polars_nyc.py' in cmd:
        return 'Polars'
    return 'Outro'

def extrair_tamanho(cmd):
    for tamanho in ['10MB', '100MB', '1GB', '1000MB']:
        if tamanho in cmd:
            return tamanho
    return 'Desconhecido'

uso_global['Biblioteca'] = uso_global['Cmdline'].apply(extrair_biblioteca)
uso_global['Arquivo'] = uso_global['Cmdline'].apply(extrair_tamanho)

# Remove processos desconhecidos ou dados incompletos
uso_global = uso_global[uso_global['Biblioteca'].isin(['Pandas', 'Polars'])]
uso_global = uso_global.dropna(subset=['Memória (MB)', 'CPU (%)', 'Timestamp'])

# Cria diretório de saída
output_dir = "graficos_por_ferramenta_tempo_execucao"
os.makedirs(output_dir, exist_ok=True)

# Gera gráficos por ferramenta e tamanho
for (biblioteca, arquivo), df_sub in uso_global.groupby(['Biblioteca', 'Arquivo']):
    # Normaliza o tempo de execução para cada processo (milissegundos desde o início)
    df_sub_sorted = df_sub.sort_values(['PID', 'Timestamp']).copy()
    df_sub_sorted['Inicio'] = df_sub_sorted.groupby('PID')['Timestamp'].transform('min')
    df_sub_sorted['Tempo Exec (ms)'] = (df_sub_sorted['Timestamp'] - df_sub_sorted['Inicio']).dt.total_seconds() * 1000

    # Agrupa e calcula médias por tempo
    df_mean = df_sub_sorted.groupby('Tempo Exec (ms)')[['CPU (%)', 'Memória (MB)']].mean().reset_index()

    # Gráficos
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sns.lineplot(data=df_mean, x='Tempo Exec (ms)', y='Memória (MB)', ax=axes[0])
    axes[0].set_title(f'Uso Médio de Memória - {biblioteca} - {arquivo}')
    axes[0].grid(True)

    sns.lineplot(data=df_mean, x='Tempo Exec (ms)', y='CPU (%)', ax=axes[1])
    axes[1].set_title(f'Uso Médio de CPU - {biblioteca} - {arquivo}')
    axes[1].grid(True)

    plt.tight_layout()
    filename = f"{biblioteca}_{arquivo}_media_por_tempo.png".replace(" ", "_")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()



############


# Cria diretório para os gráficos por PID com finalização destacada
output_dir_finais = "graficos_por_execucao_com_fim"
os.makedirs(output_dir_finais, exist_ok=True)

# Gera gráficos por combinação de Biblioteca + Arquivo
for (biblioteca, arquivo), df_sub in uso_global.groupby(['Biblioteca', 'Arquivo']):
    df_sub = df_sub.copy()
    df_sub['Inicio'] = df_sub.groupby('PID')['Timestamp'].transform('min')
    df_sub['Tempo Exec (s)'] = (df_sub['Timestamp'] - df_sub['Inicio']).dt.total_seconds()

    # Coleta tempos finais por execução
    tempos_finais = df_sub.groupby('PID')['Tempo Exec (s)'].max().reset_index()
    tempos_finais = tempos_finais.rename(columns={'Tempo Exec (s)': 'Duração (s)'})

    # Gera gráfico de CPU por tempo, uma linha por execução
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_sub, x='Tempo Exec (s)', y='CPU (%)', hue='PID', palette='tab10', legend=False)
    for _, row in tempos_finais.iterrows():
        plt.axvline(x=row['Duração (s)'], linestyle='--', color='gray', alpha=0.5)
        plt.text(row['Duração (s)'], plt.ylim()[1] * 0.95, f"{row['Duração (s)']:.1f}s",
                 rotation=90, va='top', ha='right', fontsize=8)

    plt.title(f"CPU por Execução - {biblioteca} - {arquivo}")
    plt.xlabel("Tempo de execução (s)")
    plt.ylabel("CPU (%)")
    plt.grid(True)
    plt.tight_layout()
    filename = f"{biblioteca}_{arquivo}_execucoes_individuais.png".replace(" ", "_")
    plt.savefig(os.path.join(output_dir_finais, filename))
    plt.close()


# Cria DataFrame com tempos finais por PID
df_tempos = uso_global.copy()
df_tempos['Inicio'] = df_tempos.groupby('PID')['Timestamp'].transform('min')
df_tempos['Fim'] = df_tempos.groupby('PID')['Timestamp'].transform('max')
df_tempos['Duracao (s)'] = (df_tempos['Fim'] - df_tempos['Inicio']).dt.total_seconds()

# Agrupa e remove duplicatas por PID
df_execucoes = df_tempos[['PID', 'Biblioteca', 'Arquivo', 'Duracao (s)']].drop_duplicates()

# Gera boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_execucoes, x='Arquivo', y='Duracao (s)', hue='Biblioteca')
plt.title('Tempo de Execução por Ferramenta e Tamanho de Arquivo')
plt.xlabel('Tamanho do Arquivo')
plt.ylabel('Duração da Execução (s)')
plt.grid(True)
plt.tight_layout()
plt.savefig("graficos/tempo_medio_arquivo.png")
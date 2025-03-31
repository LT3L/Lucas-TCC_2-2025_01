import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o CSV de uso global
uso_global = pd.read_csv("uso_global.csv")

# Filtrar apenas os processos que executaram scripts Python
uso_global = uso_global[uso_global['Cmdline'].str.contains('.py', na=False)].copy()

# Extração de campos úteis: nome da biblioteca (pandas/polars) e nome do arquivo
def extrair_biblioteca(cmd):
    if 'pd_nyc.py' in cmd:
        return 'Pandas'
    elif 'polars_nyc.py' in cmd:
        return 'Polars'
    return 'Outro'

def extrair_tamanho(cmd):
    for tamanho in ['10MB', '100MB', '1GB']:
        if tamanho in cmd:
            return tamanho
    return 'Desconhecido'

uso_global['Biblioteca'] = uso_global['Cmdline'].apply(extrair_biblioteca)
uso_global['Arquivo'] = uso_global['Cmdline'].apply(extrair_tamanho)

# Normalizar timestamps (convertendo para datetime para futuros gráficos temporais)
uso_global['Timestamp'] = pd.to_datetime(uso_global['Timestamp'], errors='coerce')

# Remover entradas com biblioteca desconhecida
uso_global = uso_global[uso_global['Biblioteca'].isin(['Pandas', 'Polars'])]

# Gráfico 1: Uso de memória ao longo do tempo
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=uso_global,
    x='Timestamp',
    y='Memória (MB)',
    hue='Biblioteca',
    style='Arquivo',
    markers=False
)
plt.title('Uso de Memória ao Longo do Tempo')
plt.ylabel('Memória (MB)')
plt.xlabel('Tempo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Gráfico 2: Uso de CPU ao longo do tempo
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=uso_global,
    x='Timestamp',
    y='CPU (%)',
    hue='Biblioteca',
    style='Arquivo',
    markers=False
)
plt.title('Uso de CPU ao Longo do Tempo')
plt.ylabel('CPU (%)')
plt.xlabel('Tempo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Gráfico 3: Boxplot de uso de memória por biblioteca e tamanho
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=uso_global,
    x='Arquivo',
    y='Memória (MB)',
    hue='Biblioteca'
)
plt.title('Distribuição de Uso de Memória por Biblioteca e Tamanho de Arquivo')
plt.tight_layout()
plt.grid(True)
plt.show()

# Gráfico 4: Boxplot de uso de CPU por biblioteca e tamanho
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=uso_global,
    x='Arquivo',
    y='CPU (%)',
    hue='Biblioteca'
)
plt.title('Distribuição de Uso de CPU por Biblioteca e Tamanho de Arquivo')
plt.tight_layout()
plt.grid(True)
plt.show()

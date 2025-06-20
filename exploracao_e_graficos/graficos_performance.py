import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend sem GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def carregar_dados_processados():
    """
    Carrega os dados processados do arquivo CSV gerado por metricas.py
    """
    # Tentar diferentes caminhos possíveis
    possible_paths = [
        Path("metricas_benchmark_resumo_com_formato.csv"),  # Se executado de exploracao_e_graficos
        Path("exploracao_e_graficos/metricas_benchmark_resumo_com_formato.csv"),  # Se executado da raiz
    ]
    
    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break
    
    if data_file is None:
        print("Erro: Não foi possível encontrar o arquivo de métricas processadas!")
        print("Caminhos tentados:")
        for path in possible_paths:
            print(f"  - {path.absolute()}")
        return pd.DataFrame()
    
    print(f"Carregando dados processados de: {data_file.absolute()}")
    
    try:
        df = pd.read_csv(data_file)
        print(f"Dados carregados com sucesso: {len(df)} registros")
        print("\nColunas disponíveis:")
        print(df.columns.tolist())
        print("\nPrimeiras linhas do DataFrame:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Erro ao carregar dados processados: {e}")
        return pd.DataFrame()

def criar_grafico_tempo_medio_por_biblioteca_tamanho(df):
    """
    Cria 4 gráficos de barras em layout 2x2: um para cada tamanho de dataset
    """
    # Usar os dados já processados com média em duas etapas
    tempo_medio = df.pivot_table(
        values='tempo_medio',
        index='tamanho_dataset_nominal_mb',
        columns='biblioteca'
    )
    
    # Definir tamanhos e labels
    tamanhos = sorted(df['tamanho_dataset_nominal_mb'].unique())
    labels_tamanhos = [f'{int(x)} MB' for x in tamanhos]
    
    # Criar figura com subplots em layout 2x2
    n_tamanhos = len(tamanhos)
    n_rows = (n_tamanhos + 1) // 2  # Arredonda para cima
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 8*n_rows))
    axes = axes.flatten()
    
    # Criar um subplot para cada tamanho
    for i, (tamanho, label_tamanho) in enumerate(zip(tamanhos, labels_tamanhos)):
        ax = axes[i]
        
        # Verificar se há dados para este tamanho
        if tamanho in tempo_medio.index:
            dados_tamanho = tempo_medio.loc[[tamanho]]
            
            # Criar gráfico de barras
            dados_tamanho.plot(kind='bar', ax=ax, width=0.6)
            
            # Personalizar cada subplot
            ax.set_title(f'Dataset de {label_tamanho}', 
                        fontsize=20, fontweight='bold', pad=25)
            ax.set_xlabel('Tamanho do Dataset', fontsize=16, fontweight='bold')
            ax.set_ylabel('Tempo Médio (segundos)', fontsize=16, fontweight='bold')
            
            # Configurar eixo X
            ax.set_xticklabels([label_tamanho], rotation=0, fontsize=14)
            
            # Configurar legenda
            ax.legend(title='Biblioteca', title_fontsize=14, fontsize=12, 
                     loc='upper left', frameon=True, fancybox=True, shadow=True)
            
            # Grid mais visível
            ax.grid(True, alpha=0.4, axis='y', linestyle='--')
            
            # Configurar eixo Y
            ax.tick_params(axis='y', labelsize=12)
            
            # Adicionar valores nas barras
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1fs', fontsize=11, fontweight='bold')
            
            # Ajustar espaçamento entre barras
            for patch in ax.patches:
                patch.set_width(patch.get_width() * 0.8)
        else:
            # Se não há dados, ocultar o subplot
            ax.set_visible(False)
    
    # Ocultar subplots extras se houver
    for i in range(len(tamanhos), len(axes)):
        axes[i].set_visible(False)
    
    # Título geral da figura
    fig.suptitle('Tempo Médio de Execução por Biblioteca e Tamanho de Dataset', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Ajustar espaçamento entre subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('tempo_medio_por_biblioteca.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como: tempo_medio_por_biblioteca.png")
    
    plt.close()

def criar_grafico_comparativo_escalabilidade(df):
    """
    Cria gráfico de linha mostrando como o tempo escala com o tamanho
    """
    tempo_medio = df.pivot_table(
        values='tempo_medio',
        index='tamanho_dataset_nominal_mb',
        columns='biblioteca'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotar linha para cada biblioteca
    for biblioteca in tempo_medio.columns:
        ax.plot(tempo_medio.index, 
                tempo_medio[biblioteca], 
                marker='o', linewidth=2, markersize=8, 
                label=biblioteca.capitalize())
    
    # Personalizar o gráfico
    ax.set_title('Escalabilidade: Tempo de Execução vs Tamanho do Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tamanho do Dataset (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo Médio de Execução (segundos)', fontsize=12, fontweight='bold')
    
    # Usar escala logarítmica para melhor visualização
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Personalizar eixos
    ax.grid(True, alpha=0.3)
    ax.legend(title='Biblioteca', title_fontsize=12, fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('escalabilidade_bibliotecas.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como: escalabilidade_bibliotecas.png")
    
    plt.close()

def criar_grafico_por_formato(df):
    """
    Cria gráfico de barras agrupadas por formato de arquivo
    """
    # Usar os dados já processados com média em duas etapas
    tempo_medio = df.pivot_table(
        values='tempo_medio',
        index=['tamanho_dataset_nominal_mb', 'dataset_formato'],
        columns='biblioteca'
    )
    
    # Criar subplots para cada formato
    formatos = sorted(df['dataset_formato'].unique())
    fig, axes = plt.subplots(1, len(formatos), figsize=(18, 6))
    
    if len(formatos) == 1:
        axes = [axes]
    
    for idx, formato in enumerate(formatos):
        # Filtrar dados para o formato atual
        data_formato = tempo_medio.xs(formato, level='dataset_formato')
        
        # Plotar barras
        data_formato.plot(kind='bar', ax=axes[idx], width=0.8)
        
        # Personalizar subplot
        axes[idx].set_title(f'Formato: {formato.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Tamanho (MB)', fontsize=10)
        axes[idx].set_ylabel('Tempo (s)', fontsize=10)
        axes[idx].set_xticklabels([f'{int(x)}MB' for x in data_formato.index], rotation=45)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(title='Biblioteca', fontsize=8)
    
    # Título geral
    fig.suptitle('Tempo Médio de Execução por Formato de Arquivo', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('tempo_por_formato.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como: tempo_por_formato.png")
    
    plt.close()

def criar_heatmap_performance(df):
    """
    Cria heatmap da performance (tempo médio) por biblioteca e tamanho
    """
    # Usar os dados já processados com média em duas etapas
    tempo_medio = df.pivot_table(
        values='tempo_medio',
        index='biblioteca',
        columns='tamanho_dataset_nominal_mb'
    )
    
    # Criar heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(tempo_medio, 
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd', 
                ax=ax,
                cbar_kws={'label': 'Tempo Médio (segundos)'})
    
    # Personalizar
    ax.set_title('Heatmap: Tempo Médio de Execução', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tamanho do Dataset (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Biblioteca', fontsize=12, fontweight='bold')
    
    # Personalizar labels do eixo x
    ax.set_xticklabels([f'{int(x)}MB' for x in tempo_medio.columns], rotation=0)
    ax.set_yticklabels([x.capitalize() for x in tempo_medio.index], rotation=0)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('heatmap_performance.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como: heatmap_performance.png")
    
    plt.close()

def criar_grafico_memoria_por_formato(df):
    """
    Cria gráfico de barras agrupadas mostrando o uso médio de memória por formato e tamanho do arquivo
    """
    # Calcular a média de memória por formato e tamanho, independente da biblioteca
    memoria_media = df.groupby(['dataset_formato', 'tamanho_dataset_nominal_mb'])['memoria_media'].mean().reset_index()
    
    # Criar subplots para cada formato
    formatos = sorted(df['dataset_formato'].unique())
    fig, axes = plt.subplots(1, len(formatos), figsize=(18, 6))
    
    if len(formatos) == 1:
        axes = [axes]
    
    for idx, formato in enumerate(formatos):
        # Filtrar dados para o formato atual
        data_formato = memoria_media[memoria_media['dataset_formato'] == formato]
        
        # Ordenar por tamanho para garantir ordem correta das barras
        data_formato = data_formato.sort_values('tamanho_dataset_nominal_mb')
        
        # Plotar barras
        ax = axes[idx]
        bars = ax.bar(range(len(data_formato)), 
                     data_formato['memoria_media'],
                     width=0.6)
        
        # Personalizar subplot
        ax.set_title(f'Formato: {formato.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tamanho do Arquivo', fontsize=10)
        ax.set_ylabel('Memória Média (MB)', fontsize=10)
        
        # Configurar eixo X com labels de tamanho
        ax.set_xticks(range(len(data_formato)))
        ax.set_xticklabels([f'{int(x)}MB' for x in data_formato['tamanho_dataset_nominal_mb']], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} MB',
                   ha='center', va='bottom', fontsize=8)
    
    # Título geral
    fig.suptitle('Uso Médio de Memória por Formato e Tamanho do Arquivo', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('memoria_por_formato.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como: memoria_por_formato.png")
    
    plt.close()

def exibir_estatisticas_resumo(df):
    """
    Exibe estatísticas resumo dos dados
    """
    print("\n" + "="*60)
    print("ESTATÍSTICAS RESUMO DOS GRÁFICOS")
    print("="*60)
    
    print(f"Total de combinações analisadas: {len(df)}")
    print(f"Bibliotecas: {', '.join(sorted(df['biblioteca'].unique()))}")
    print(f"Tamanhos de dataset: {', '.join([f'{x}MB' for x in sorted(df['tamanho_dataset_nominal_mb'].unique())])}")
    print(f"Formatos: {', '.join(sorted(df['dataset_formato'].unique()))}")
    
    # Tempo médio geral por biblioteca
    tempo_por_lib = df.groupby('biblioteca')['tempo_medio'].mean().sort_values()
    print(f"\nTempo médio geral por biblioteca:")
    for lib, tempo in tempo_por_lib.items():
        print(f"  {lib.capitalize()}: {tempo:.2f}s")
    
    # Biblioteca mais rápida por tamanho
    print(f"\nBiblioteca mais rápida por tamanho:")
    for tamanho in sorted(df['tamanho_dataset_nominal_mb'].unique()):
        data_tamanho = df[df['tamanho_dataset_nominal_mb'] == tamanho]
        mais_rapida = data_tamanho.groupby('biblioteca')['tempo_medio'].mean().idxmin()
        tempo_min = data_tamanho.groupby('biblioteca')['tempo_medio'].mean().min()
        print(f"  {tamanho}MB: {mais_rapida.capitalize()} ({tempo_min:.2f}s)")

def main():
    """
    Função principal
    """
    print("Iniciando geração de gráficos de performance...")
    
    # Carregar dados processados
    df = carregar_dados_processados()
    if df.empty:
        print("Nenhum dado encontrado!")
        return
    
    # Exibir estatísticas
    exibir_estatisticas_resumo(df)
    
    print("\nGerando gráficos...")
    
    # Gerar todos os gráficos
    try:
        criar_grafico_tempo_medio_por_biblioteca_tamanho(df)
        criar_grafico_comparativo_escalabilidade(df)
        criar_grafico_por_formato(df)
        criar_heatmap_performance(df)
        criar_grafico_memoria_por_formato(df)
        
        print("\n" + "="*60)
        print("GRÁFICOS GERADOS COM SUCESSO!")
        print("="*60)
        print("Arquivos salvos:")
        print("- tempo_medio_por_biblioteca.png")
        print("- escalabilidade_bibliotecas.png") 
        print("- tempo_por_formato.png")
        print("- heatmap_performance.png")
        print("- memoria_por_formato.png")
        
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")

if __name__ == "__main__":
    main() 
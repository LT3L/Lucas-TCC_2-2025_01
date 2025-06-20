import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def carregar_e_filtrar_dados():
    """Carrega e filtra dados de benchmark"""
    # Encontrar diretório de dados
    possible_paths = [
        Path("../app/datasets_and_models_output"),
        Path("app/datasets_and_models_output"),
        Path("./datasets_and_models_output"),
    ]
    
    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = path
            break
    
    if data_dir is None:
        print("Erro: Diretório de dados não encontrado!")
        return pd.DataFrame()
    
    # Carregar todos os arquivos CSV
    benchmark_files = glob.glob(str(data_dir / "benchmark_*.csv"))
    all_data = []
    
    for file in benchmark_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combinar e filtrar dados
    df = pd.concat(all_data, ignore_index=True)
    df_filtrado = df[
        (df['status'] == 'completed') & 
        (df['memoria_media_execucao'].notna()) & 
        (df['memoria_media_execucao'] > 0)
    ].copy()
    
    return df_filtrado

def criar_grafico_memoria_por_formato_biblioteca(df):
    """Cria gráfico de memória utilizada por formato e biblioteca"""
    
    # Calcular memória média por biblioteca e formato
    memoria_media = df.groupby(['biblioteca', 'dataset_formato'])['memoria_media_execucao'].mean().reset_index()
    
    # Converter de MB para GB para melhor legibilidade
    memoria_media['memoria_gb'] = memoria_media['memoria_media_execucao'] / 1024
    
    # Criar tabela pivotada
    pivot_data = memoria_media.pivot(
        index='dataset_formato', 
        columns='biblioteca', 
        values='memoria_gb'
    )
    
    # Definir formatos e cores
    formatos = sorted(df['dataset_formato'].unique())
    cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Criar figura compacta para documento
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Criar gráfico de barras agrupadas
    pivot_data.plot(kind='bar', ax=ax, width=0.8, color=cores)
    
    # Personalizar o gráfico
    ax.set_title('Memória Média Utilizada por Formato de Arquivo e Biblioteca', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Formato de Arquivo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Memória Média (GB)', fontsize=14, fontweight='bold')
    
    # Configurar eixo X
    ax.set_xticklabels([formato.upper() for formato in pivot_data.index], rotation=0, fontsize=12)
    
    # Configurar legenda
    ax.legend(title='Biblioteca', title_fontsize=12, fontsize=11, 
             loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Grid sutil
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    
    # Configurar eixo Y
    ax.tick_params(axis='y', labelsize=11)
    
    # Adicionar valores nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f GB', fontsize=10, fontweight='bold', rotation=90)
    
    # Ajustar limites
    ax.margins(x=0.1)
    
    # Remover spines desnecessários
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar com alta qualidade
    plt.savefig('grafico_memoria_por_formato.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print("Gráfico salvo como: grafico_memoria_por_formato.png")
    
    # Exibir dados da tabela
    print("\nMemória Média por Formato e Biblioteca (GB):")
    print("=" * 60)
    print(pivot_data.round(2))
    
    return pivot_data

def criar_grafico_memoria_por_tamanho_biblioteca(df):
    """Cria gráfico de memória por tamanho de dataset e biblioteca"""
    
    # Calcular memória média por biblioteca e tamanho
    memoria_media = df.groupby(['biblioteca', 'tamanho_dataset_nominal_mb'])['memoria_media_execucao'].mean().reset_index()
    
    # Converter para GB
    memoria_media['memoria_gb'] = memoria_media['memoria_media_execucao'] / 1024
    
    # Criar tabela pivotada
    pivot_data = memoria_media.pivot(
        index='tamanho_dataset_nominal_mb', 
        columns='biblioteca', 
        values='memoria_gb'
    )
    
    # Definir tamanhos e labels
    tamanhos = [100, 1000, 10000, 50000]
    labels_tamanhos = ['100 MB', '1 GB', '10 GB', '50 GB']
    cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Criar figura com layout 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Criar um subplot para cada tamanho
    for i, (tamanho, label_tamanho) in enumerate(zip(tamanhos, labels_tamanhos)):
        ax = axes[i]
        
        # Verificar se há dados para este tamanho
        if tamanho in pivot_data.index:
            dados_tamanho = pivot_data.loc[[tamanho]]
            
            # Criar gráfico de barras
            dados_tamanho.plot(kind='bar', ax=ax, width=0.75, color=cores)
            
            # Personalizar cada subplot
            ax.set_title(f'{label_tamanho}', fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel('')
            ax.set_ylabel('Memória (GB)', fontsize=14, fontweight='bold')
            
            # Remover labels do eixo X
            ax.set_xticklabels([])
            ax.set_xticks([])
            
            # Configurar legenda apenas no primeiro gráfico
            if i == 0:
                ax.legend(title='Biblioteca', title_fontsize=12, fontsize=10, 
                         loc='upper right', frameon=True, fancybox=True, shadow=True)
            else:
                ax.legend().set_visible(False)
            
            # Grid sutil
            ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='y', labelsize=11, pad=2)
            
            # Adicionar valores nas barras
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f GB', fontsize=10, fontweight='bold', padding=2)
            
            # Ajustar limites
            ax.margins(x=0.15)
            
            # Remover spines desnecessários
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
        else:
            ax.set_visible(False)
    
    # Título geral
    fig.suptitle('Memória Média Utilizada por Biblioteca e Tamanho de Dataset', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Ajustar espaçamento
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, 
                       wspace=0.25, hspace=0.35)
    
    # Salvar
    plt.savefig('grafico_memoria_por_tamanho.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print("Gráfico salvo como: grafico_memoria_por_tamanho.png")
    
    return pivot_data

def analisar_memoria_detalhada(df):
    """Análise detalhada do uso de memória"""
    
    print("\n" + "="*70)
    print("ANÁLISE DETALHADA DE USO DE MEMÓRIA")
    print("="*70)
    
    # Estatísticas gerais
    print(f"Total de execuções com dados de memória: {len(df)}")
    
    # Memória média por biblioteca
    memoria_por_lib = df.groupby('biblioteca')['memoria_media_execucao'].mean() / 1024
    print(f"\nMemória média por biblioteca (GB):")
    for lib, memoria in memoria_por_lib.sort_values().items():
        print(f"  {lib.capitalize()}: {memoria:.2f} GB")
    
    # Memória por formato
    memoria_por_formato = df.groupby('dataset_formato')['memoria_media_execucao'].mean() / 1024
    print(f"\nMemória média por formato (GB):")
    for formato, memoria in memoria_por_formato.sort_values().items():
        print(f"  {formato.upper()}: {memoria:.2f} GB")
    
    # Maior e menor consumo
    max_memoria = df.loc[df['memoria_media_execucao'].idxmax()]
    min_memoria = df.loc[df['memoria_media_execucao'].idxmin()]
    
    print(f"\nMaior consumo de memória:")
    print(f"  {max_memoria['biblioteca'].capitalize()} - {max_memoria['dataset_formato'].upper()} - {max_memoria['tamanho_dataset_nominal_mb']}MB: {max_memoria['memoria_media_execucao']/1024:.2f} GB")
    
    print(f"\nMenor consumo de memória:")
    print(f"  {min_memoria['biblioteca'].capitalize()} - {min_memoria['dataset_formato'].upper()} - {min_memoria['tamanho_dataset_nominal_mb']}MB: {min_memoria['memoria_media_execucao']/1024:.2f} GB")
    
    # Eficiência de memória (menor é melhor)
    print(f"\nRanking de eficiência de memória (menor consumo):")
    ranking = memoria_por_lib.sort_values()
    for i, (lib, memoria) in enumerate(ranking.items(), 1):
        print(f"  {i}º {lib.capitalize()}: {memoria:.2f} GB")

def main():
    """Função principal"""
    print("Gerando gráficos de uso de memória...")
    
    # Carregar dados
    df = carregar_e_filtrar_dados()
    if df.empty:
        print("Nenhum dado válido encontrado!")
        return
    
    # Verificar se há dados de memória
    if 'memoria_media_execucao' not in df.columns:
        print("Coluna de memória não encontrada nos dados!")
        return
    
    print(f"Dados carregados: {len(df)} execuções com dados de memória")
    
    # Gerar gráficos
    print("\nGerando gráfico de memória por formato...")
    pivot_formato = criar_grafico_memoria_por_formato_biblioteca(df)
    
    print("\nGerando gráfico de memória por tamanho...")
    pivot_tamanho = criar_grafico_memoria_por_tamanho_biblioteca(df)
    
    # Análise detalhada
    analisar_memoria_detalhada(df)
    
    print("\n" + "="*60)
    print("GRÁFICOS DE MEMÓRIA GERADOS COM SUCESSO!")
    print("="*60)
    print("Arquivos salvos:")
    print("- grafico_memoria_por_formato.png")
    print("- grafico_memoria_por_tamanho.png")

if __name__ == "__main__":
    main() 
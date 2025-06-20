import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
from pathlib import Path

def carregar_e_filtrar_dados():
    """Carrega dados do arquivo de benchmark filtrado"""
    arquivo_filtrado = "dados_benchmark_filtrados.csv"
    
    try:
        df = pd.read_csv(arquivo_filtrado)
        print(f"Dados carregados com sucesso do arquivo: {arquivo_filtrado}")
        return df
    except Exception as e:
        print(f"Erro ao carregar {arquivo_filtrado}: {e}")
        return pd.DataFrame()

def criar_grafico_barras_simples(df):
    """Cria 4 gráficos de barras em layout 2x2: um para cada tamanho de dataset - otimizado para documentos"""
    
    # Calcular tempo médio por biblioteca e tamanho
    tempo_medio = df.groupby(['biblioteca', 'tamanho_dataset_nominal_mb'])['tempo_execucao'].mean().reset_index()
    
    # Criar tabela pivotada
    pivot_data = tempo_medio.pivot(
        index='tamanho_dataset_nominal_mb', 
        columns='biblioteca', 
        values='tempo_execucao'
    )
    
    # Definir tamanhos e labels
    tamanhos = [100, 1000, 10000, 50000]
    labels_tamanhos = ['100 MB', '1 GB', '10 GB', '50 GB']
    
    # Cores consistentes para as bibliotecas
    cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Criar figura com 4 subplots em layout 2x2 - mais compacta
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Converter para array 1D para facilitar iteração
    
    # Criar um subplot para cada tamanho
    for i, (tamanho, label_tamanho) in enumerate(zip(tamanhos, labels_tamanhos)):
        ax = axes[i]
        
        # Verificar se há dados para este tamanho
        if tamanho in pivot_data.index:
            dados_tamanho = pivot_data.loc[[tamanho]]
            
            # Criar gráfico de barras com barras mais largas
            dados_tamanho.plot(kind='bar', ax=ax, width=0.75, color=cores)
            
            # Personalizar cada subplot
            ax.set_title(f'{label_tamanho}', fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel('')  # Remover label do eixo X para economizar espaço
            ax.set_ylabel('Tempo (s)', fontsize=14, fontweight='bold')
            
            # Remover labels do eixo X para economizar espaço
            ax.set_xticklabels([])
            ax.set_xticks([])
            
            # Configurar legenda apenas no primeiro gráfico
            if i == 0:
                ax.legend(title='Biblioteca', title_fontsize=12, fontsize=10, 
                         loc='upper right', frameon=True, fancybox=True, shadow=True,
                         bbox_to_anchor=(1.0, 1.0))
            else:
                ax.legend().set_visible(False)  # Ocultar legendas dos outros gráficos
            
            # Grid mais sutil
            ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            
            # Configurar eixo Y com menos espaço
            ax.tick_params(axis='y', labelsize=11, pad=2)
            
            # Adicionar valores nas barras
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1fs', fontsize=10, fontweight='bold', padding=2)
            
            # Ajustar limites para reduzir espaço em branco
            ax.margins(x=0.15)  # Reduzir margens laterais
            
            # Remover spines desnecessários
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
        else:
            # Se não há dados, ocultar o subplot
            ax.set_visible(False)
    
    # Título geral da figura mais compacto
    fig.suptitle('Tempo Médio de Execução por Biblioteca e Tamanho de Dataset', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Ajustar espaçamento entre subplots para ser mais compacto
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, 
                       wspace=0.25, hspace=0.35)
    
    # Salvar com alta qualidade e compacto
    plt.savefig('grafico_tempo_medio_simples.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print("Gráfico salvo como: grafico_tempo_medio_simples.png")
    
    # Exibir dados organizados por tamanho
    print("\nDados do gráfico por tamanho:")
    print("=" * 70)
    
    for tamanho, label in zip(tamanhos, labels_tamanhos):
        if tamanho in pivot_data.index:
            print(f"\n{label}:")
            dados_linha = pivot_data.loc[tamanho]
            for biblioteca in dados_linha.index:
                if pd.notna(dados_linha[biblioteca]):
                    print(f"  {biblioteca.capitalize()}: {dados_linha[biblioteca]:.2f}s")
    
    # Mostrar estatísticas
    print(f"\nResumo:")
    print(f"- Total de execuções analisadas: {len(df)}")
    print(f"- Bibliotecas: {', '.join(sorted(df['biblioteca'].unique()))}")
    print(f"- Tamanhos: {', '.join([f'{x}MB' for x in sorted(df['tamanho_dataset_nominal_mb'].unique())])}")
    
    # Biblioteca mais rápida por tamanho
    print(f"\nBiblioteca mais rápida por tamanho:")
    for tamanho, label in zip(tamanhos, labels_tamanhos):
        if tamanho in pivot_data.index:
            melhor_lib = pivot_data.loc[tamanho].idxmin()
            melhor_tempo = pivot_data.loc[tamanho].min()
            print(f"  {label}: {melhor_lib.capitalize()} ({melhor_tempo:.2f}s)")
    
    # Análise de speedup mais detalhada
    print(f"\nAnálise de Speedup (Polars vs outros):")
    for tamanho, label in zip(tamanhos, labels_tamanhos):
        if tamanho in pivot_data.index and 'polars' in pivot_data.columns:
            polars_tempo = pivot_data.loc[tamanho, 'polars']
            if pd.notna(polars_tempo):
                print(f"\n  {label}:")
                for lib in pivot_data.columns:
                    if lib != 'polars' and pd.notna(pivot_data.loc[tamanho, lib]):
                        speedup = pivot_data.loc[tamanho, lib] / polars_tempo
                        economia_tempo = pivot_data.loc[tamanho, lib] - polars_tempo
                        print(f"    vs {lib.capitalize()}: {speedup:.2f}x mais rápido (economia de {economia_tempo:.1f}s)")
    
    # Resumo de performance geral
    print(f"\nResumo de Performance:")
    tempo_total_por_lib = df.groupby('biblioteca')['tempo_execucao'].mean().sort_values()
    print("Ranking geral (tempo médio):")
    for i, (lib, tempo) in enumerate(tempo_total_por_lib.items(), 1):
        print(f"  {i}º {lib.capitalize()}: {tempo:.2f}s")

def main():
    """Função principal"""
    print("Gerando gráfico simples de tempo médio por biblioteca e tamanho...")
    
    # Carregar dados
    df = carregar_e_filtrar_dados()
    if df.empty:
        print("Nenhum dado válido encontrado!")
        return
    
    # Criar gráfico
    criar_grafico_barras_simples(df)
    
    print("\nGráfico gerado com sucesso!")

if __name__ == "__main__":
    main() 
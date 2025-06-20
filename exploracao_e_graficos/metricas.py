import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np

def carregar_todos_benchmarks():
    """
    Carrega todos os arquivos de benchmark CSV disponíveis
    """
    # Tentar diferentes caminhos possíveis
    possible_paths = [
        Path("../app/datasets_and_models_output"),  # Se executado de exploracao_e_graficos
        Path("app/datasets_and_models_output"),     # Se executado da raiz do projeto
        Path("./datasets_and_models_output"),       # Se executado do diretório app
    ]
    
    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = path
            break
    
    if data_dir is None:
        print("Erro: Não foi possível encontrar o diretório de dados!")
        print("Caminhos tentados:")
        for path in possible_paths:
            print(f"  - {path.absolute()}")
        return pd.DataFrame()
    
    print(f"Usando diretório de dados: {data_dir.absolute()}")
    
    # Buscar todos os arquivos CSV de benchmark
    benchmark_files = glob.glob(str(data_dir / "benchmark_*.csv"))
    
    print(f"Encontrados {len(benchmark_files)} arquivos de benchmark")
    
    # Lista para armazenar todos os dataframes
    all_data = []
    
    for file in benchmark_files:
        try:
            df = pd.read_csv(file)
            print(f"Carregado: {os.path.basename(file)} - {len(df)} registros")
            all_data.append(df)
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")
    
    # Concatenar todos os dataframes
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal de registros combinados: {len(combined_df)}")
        return combined_df
    else:
        print("Nenhum dado foi carregado!")
        return pd.DataFrame()

def filtrar_execucoes_bem_sucedidas(df):
    """
    Filtra apenas execuções que foram bem sucedidas
    """
    print(f"Total de registros antes do filtro: {len(df)}")
    
    # Verificar valores únicos na coluna status
    print(f"Status únicos encontrados: {df['status'].unique()}")
    
    # Filtrar apenas execuções completadas
    df_filtrado = df[df['status'] == 'completed'].copy()
    
    print(f"Registros após filtrar apenas 'completed': {len(df_filtrado)}")
    
    # Remover registros com tempo de execução inválido (NaN ou <= 0)
    df_filtrado = df_filtrado[
        (df_filtrado['tempo_execucao'].notna()) & 
        (df_filtrado['tempo_execucao'] > 0)
    ]
    
    print(f"Registros após filtrar tempos válidos: {len(df_filtrado)}")
    
    return df_filtrado

def calcular_metricas_por_tamanho_e_ferramenta(df):
    """
    Calcula métricas agrupadas por tamanho de dataset, ferramenta e formato de arquivo
    """
    print("\n=== ANÁLISE DE MÉTRICAS POR TAMANHO, FERRAMENTA E FORMATO ===")
    
    # Agrupar por biblioteca, tamanho de dataset e formato
    grouped = df.groupby(['biblioteca', 'tamanho_dataset_nominal_mb', 'dataset_formato'])
    
    # Calcular estatísticas
    metricas = grouped.agg({
        'tempo_execucao': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'cpu_medio_execucao': ['mean'],
        'memoria_media_execucao': ['mean']
    }).round(4)
    
    # Simplificar nomes das colunas
    metricas.columns = [
        'num_execucoes', 'tempo_medio', 'tempo_mediano', 'tempo_desvio', 
        'tempo_min', 'tempo_max', 'cpu_medio', 'memoria_media'
    ]
    
    return metricas

def exibir_relatorio_detalhado(df, metricas):
    """
    Exibe relatório detalhado das métricas
    """
    print("\n" + "="*80)
    print("RELATÓRIO DE MÉTRICAS DE BENCHMARK")
    print("="*80)
    
    print(f"\nTotal de execuções bem-sucedidas analisadas: {len(df)}")
    print(f"Bibliotecas: {sorted(df['biblioteca'].unique())}")
    print(f"Tamanhos de dataset (MB): {sorted(df['tamanho_dataset_nominal_mb'].unique())}")
    print(f"Datasets: {sorted(df['dataset_nome'].unique())}")
    print(f"Formatos: {sorted(df['dataset_formato'].unique())}")
    
    print("\n" + "-"*80)
    print("MÉDIAS DE TEMPO DE EXECUÇÃO POR BIBLIOTECA E TAMANHO (segundos)")
    print("-"*80)
    
    # Primeiro passo: média por dataset para cada tamanho
    df_dataset_avg = df.groupby(['biblioteca', 'tamanho_dataset_nominal_mb', 'dataset_nome'])['tempo_execucao'].mean().reset_index()
    
    # Segundo passo: média dos datasets para cada tamanho
    pivot_tempo = df_dataset_avg.pivot_table(
        values='tempo_execucao',
        index='biblioteca',
        columns='tamanho_dataset_nominal_mb',
        aggfunc='mean'
    ).round(4)
    
    print(pivot_tempo)
    
    print("\n" + "-"*80)
    print("MÉDIAS DE TEMPO POR FORMATO DE ARQUIVO (segundos)")
    print("-"*80)
    
    # Análise por formato com média em duas etapas
    for formato in sorted(df['dataset_formato'].unique()):
        print(f"\n=== FORMATO: {formato.upper()} ===")
        df_formato = df[df['dataset_formato'] == formato]
        
        # Primeiro passo: média por dataset para cada tamanho
        df_formato_dataset_avg = df_formato.groupby(['biblioteca', 'tamanho_dataset_nominal_mb', 'dataset_nome'])['tempo_execucao'].mean().reset_index()
        
        # Segundo passo: média dos datasets para cada tamanho
        pivot_formato = df_formato_dataset_avg.pivot_table(
            values='tempo_execucao',
            index='biblioteca',
            columns='tamanho_dataset_nominal_mb',
            aggfunc='mean'
        ).round(4)
        
        print(pivot_formato)
    
    print("\n" + "-"*80)
    print("NÚMERO DE EXECUÇÕES POR BIBLIOTECA, TAMANHO E FORMATO")
    print("-"*80)
    
    pivot_count = df.pivot_table(
        values='tempo_execucao',
        index=['biblioteca', 'dataset_formato'],
        columns='tamanho_dataset_nominal_mb',
        aggfunc='count'
    )
    
    print(pivot_count)
    
    print("\n" + "-"*80)
    print("MÉTRICAS DETALHADAS (POR BIBLIOTECA, TAMANHO E FORMATO)")
    print("-"*80)
    print(metricas)
    
    # Comparação de performance por formato com média em duas etapas
    print("\n" + "-"*80)
    print("COMPARAÇÃO DE PERFORMANCE (PANDAS vs POLARS) POR FORMATO")
    print("-"*80)
    
    for formato in sorted(df['dataset_formato'].unique()):
        print(f"\n=== FORMATO: {formato.upper()} ===")
        df_formato = df[df['dataset_formato'] == formato]
        
        for tamanho in sorted(df_formato['tamanho_dataset_nominal_mb'].unique()):
            print(f"\nDataset {tamanho}MB ({formato}):")
            
            # Primeiro passo: média por dataset
            df_tamanho = df_formato[df_formato['tamanho_dataset_nominal_mb'] == tamanho]
            df_dataset_avg = df_tamanho.groupby(['biblioteca', 'dataset_nome'])['tempo_execucao'].mean().reset_index()
            
            # Segundo passo: média dos datasets
            pandas_tempo = df_dataset_avg[df_dataset_avg['biblioteca'] == 'pandas']['tempo_execucao'].mean()
            polars_tempo = df_dataset_avg[df_dataset_avg['biblioteca'] == 'polars']['tempo_execucao'].mean()
            
            if pd.notna(pandas_tempo) and pd.notna(polars_tempo):
                if pandas_tempo > polars_tempo:
                    speedup = pandas_tempo / polars_tempo
                    print(f"  Polars é {speedup:.2f}x mais rápido que Pandas")
                    print(f"  Pandas: {pandas_tempo:.4f}s | Polars: {polars_tempo:.4f}s")
                else:
                    speedup = polars_tempo / pandas_tempo
                    print(f"  Pandas é {speedup:.2f}x mais rápido que Polars")
                    print(f"  Pandas: {pandas_tempo:.4f}s | Polars: {polars_tempo:.4f}s")
            elif pd.notna(pandas_tempo) and pd.isna(polars_tempo):
                print(f"  Apenas Pandas: {pandas_tempo:.4f}s")
            elif pd.isna(pandas_tempo) and pd.notna(polars_tempo):
                print(f"  Apenas Polars: {polars_tempo:.4f}s")
            else:
                print(f"  Nenhum dado disponível")
    
    # Análise de performance por formato com média em duas etapas
    print("\n" + "-"*80)
    print("RANKING DE PERFORMANCE POR FORMATO (média geral)")
    print("-"*80)
    
    # Primeiro passo: média por dataset para cada tamanho
    df_dataset_avg = df.groupby(['dataset_formato', 'biblioteca', 'tamanho_dataset_nominal_mb', 'dataset_nome'])['tempo_execucao'].mean().reset_index()
    
    # Segundo passo: média dos datasets para cada formato e biblioteca
    formato_performance = df_dataset_avg.groupby(['dataset_formato', 'biblioteca'])['tempo_execucao'].mean().round(4)
    print(formato_performance.unstack(level=1))
    
    # Melhor formato por biblioteca com média em duas etapas
    print("\n" + "-"*80)
    print("MELHOR FORMATO POR BIBLIOTECA (menor tempo médio)")
    print("-"*80)
    
    for biblioteca in sorted(df['biblioteca'].unique()):
        df_lib = df[df['biblioteca'] == biblioteca]
        
        # Primeiro passo: média por dataset para cada formato
        df_lib_dataset_avg = df_lib.groupby(['dataset_formato', 'dataset_nome'])['tempo_execucao'].mean().reset_index()
        
        # Segundo passo: média dos datasets para cada formato
        melhor_formato = df_lib_dataset_avg.groupby('dataset_formato')['tempo_execucao'].mean().idxmin()
        melhor_tempo = df_lib_dataset_avg.groupby('dataset_formato')['tempo_execucao'].mean().min()
        print(f"{biblioteca.capitalize()}: {melhor_formato} ({melhor_tempo:.4f}s médio)")

def salvar_metricas_csv(metricas, df):
    """
    Salva as métricas em arquivo CSV
    """
    output_file = "metricas_benchmark_resumo_com_formato.csv"
    metricas.to_csv(output_file)
    print(f"\nMétricas salvas em: {output_file}")
    
    # Salvar também dados detalhados filtrados
    detailed_file = "dados_benchmark_filtrados.csv"
    colunas_importantes = [
        'biblioteca', 'dataset_nome', 'dataset_formato', 'tamanho_dataset_nominal_mb',
        'tempo_execucao', 'cpu_medio_execucao', 'memoria_media_execucao', 'status'
    ]
    df_filtrado = df[df['status'] == 'completed'].copy()
    df_filtrado[colunas_importantes].to_csv(detailed_file, index=False)
    print(f"Dados detalhados filtrados salvos em: {detailed_file}")
    
    # Salvar também resumo por formato
    formato_resumo = df_filtrado.groupby(['biblioteca', 'dataset_formato', 'tamanho_dataset_nominal_mb']).agg({
        'tempo_execucao': ['count', 'mean', 'std'],
        'cpu_medio_execucao': 'mean',
        'memoria_media_execucao': 'mean'
    }).round(4)
    
    formato_resumo.columns = ['num_execucoes', 'tempo_medio', 'tempo_desvio', 'cpu_medio', 'memoria_media']
    formato_resumo.to_csv("resumo_por_formato.csv")
    print(f"Resumo por formato salvo em: resumo_por_formato.csv")

def main():
    """
    Função principal
    """
    print("Iniciando análise de métricas de benchmark...")
    
    # Carregar todos os benchmarks
    df_completo = carregar_todos_benchmarks()
    
    if df_completo.empty:
        print("Nenhum dado para analisar!")
        return
    
    # Filtrar apenas execuções bem-sucedidas
    df_filtrado = filtrar_execucoes_bem_sucedidas(df_completo)
    
    if df_filtrado.empty:
        print("Nenhuma execução bem-sucedida encontrada!")
        return
    
    # Calcular métricas
    metricas = calcular_metricas_por_tamanho_e_ferramenta(df_filtrado)
    
    # Exibir relatório
    exibir_relatorio_detalhado(df_filtrado, metricas)
    
    # Salvar resultados
    salvar_metricas_csv(metricas, df_filtrado)
    
    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()

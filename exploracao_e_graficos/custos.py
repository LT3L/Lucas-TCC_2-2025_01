import pandas as pd
import numpy as np
import glob
import os

# Get all benchmark files
benchmark_files = glob.glob('app/datasets_and_models_output/benchmark_*.csv')

# Read and combine all benchmark files
dfs = []
for file in benchmark_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {str(e)}")

# Combine all dataframes
df = pd.concat(dfs, ignore_index=True)

# Filter only completed executions
df = df[df['status'] == 'completed']

# Calculate costs
# For 8 cores machine (cost per hour = 0.363 USD)
# For 32 cores machine (cost per hour = 1.453 USD)
df['custo_execucao'] = np.where(
    df['nucleos_logicos'] == 8,
    (df['tempo_execucao'] / 3600) * 0.363,  # Convert seconds to hours and multiply by hourly cost
    (df['tempo_execucao'] / 3600) * 1.453
)

# First calculate average by dataset, format and size
print("\nMédia de custo por dataset, formato e tamanho:")
avg_by_dataset = df.groupby(['dataset_nome', 'dataset_formato', 'tamanho_dataset_nominal_mb', 'biblioteca', 'nucleos_logicos'])['custo_execucao'].mean().reset_index()
print(avg_by_dataset.to_string(index=False))

# Now calculate averages by format, size and library using the dataset averages
print("\nMédia de custo por formato, tamanho e biblioteca (média das médias por dataset):")
avg_by_format = avg_by_dataset.groupby(['dataset_formato', 'tamanho_dataset_nominal_mb', 'biblioteca'])['custo_execucao'].mean().reset_index()
print(avg_by_format.to_string(index=False))

# Calculate averages by library, size and machine type using the dataset averages
print("\nMédia de custo por biblioteca, tamanho e tipo de máquina (média das médias por dataset):")
avg_by_machine = avg_by_dataset.groupby(['biblioteca', 'tamanho_dataset_nominal_mb', 'nucleos_logicos'])['custo_execucao'].mean().reset_index()
print(avg_by_machine.to_string(index=False))

# Calculate averages by machine, format and size using the dataset averages
print("\nMédia de custo por máquina, biblioteca, formato e tamanho (média das médias por dataset):")
avg_by_machine_format = avg_by_dataset.groupby(['nucleos_logicos', 'biblioteca', 'dataset_formato', 'tamanho_dataset_nominal_mb'])['custo_execucao'].mean().reset_index()
print(avg_by_machine_format.to_string(index=False))

# Save results to CSV
avg_by_dataset.to_csv('app/datasets_and_models_output/custos_por_dataset.csv', index=False)
avg_by_format.to_csv('app/datasets_and_models_output/custos_por_formato.csv', index=False)
avg_by_machine.to_csv('app/datasets_and_models_output/custos_por_maquina.csv', index=False)
avg_by_machine_format.to_csv('app/datasets_and_models_output/custos_por_maquina_biblioteca_formato.csv', index=False)

# Print some statistics about the data
print("\nEstatísticas sobre os dados:")
print(f"Total de arquivos de benchmark processados: {len(benchmark_files)}")
print(f"Total de execuções completadas: {len(df)}")
print("\nDistribuição por biblioteca:")
print(df['biblioteca'].value_counts())
print("\nDistribuição por formato:")
print(df['dataset_formato'].value_counts())
print("\nDistribuição por tamanho:")
print(df['tamanho_dataset_nominal_mb'].value_counts().sort_index())
print("\nDistribuição por dataset:")
print(df['dataset_nome'].value_counts())

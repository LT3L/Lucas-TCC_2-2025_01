import pandas as pd
import psutil
import platform
import joblib
import numpy as np
import os
import warnings
from pathlib import Path
import re

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', module='sklearn.utils.validation')

# Carregamento dos modelos campeões
MODELS_DIR = "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets_and_models_output/models/"

# Modelo de predição de sucesso
SUCCESS_MODEL_PATH = os.path.join(MODELS_DIR, "champion_success_predictor.pkl")
SUCCESS_THRESHOLD_PATH = os.path.join(MODELS_DIR, "champion_success_threshold.pkl")
SUCCESS_FEATURES_PATH = os.path.join(MODELS_DIR, "ensemble_memory_features.pkl")

# Modelo de estimativa de tempo
TIME_MODEL_PATH = os.path.join(MODELS_DIR, "champion_time_estimator.pkl")
TIME_FEATURES_PATH = os.path.join(MODELS_DIR, "recomender_features.pkl")

# Caminho para dados de preços da Azure
AZURE_PRICING_PATH = os.path.join(os.path.dirname(__file__), "azure_east_us_dasv6_pricing_hour.csv")

# Carregar modelos e features
print("Carregando modelos campeões...")
success_model = joblib.load(SUCCESS_MODEL_PATH)
success_threshold = joblib.load(SUCCESS_THRESHOLD_PATH)
success_features = joblib.load(SUCCESS_FEATURES_PATH)

time_model = joblib.load(TIME_MODEL_PATH)
time_features = joblib.load(TIME_FEATURES_PATH)

print("✅ Modelos campeões carregados com sucesso!")

def carregar_precos_azure():
    """Carrega os dados de preços da Azure."""
    try:
        df_precos = pd.read_csv(AZURE_PRICING_PATH)
        
        # Processar dados de RAM (converter "X GiB" para GB numérico)
        df_precos['RAM_GB'] = df_precos['RAM'].str.extract(r'(\d+)').astype(int)
        
        # Processar preços pay-as-you-go (remover $ e /hour)
        df_precos['Preco_por_hora_USD'] = df_precos['Pay as you go'].str.replace(r'[\$\/hour]', '', regex=True).astype(float)
        
        return df_precos
    except Exception as e:
        print(f"⚠️ Erro ao carregar preços da Azure: {e}")
        return None

def encontrar_vm_mais_proxima(cpu_cores, memoria_gb, df_precos):
    """Encontra a VM da Azure mais próxima baseada na configuração do sistema."""
    if df_precos is None:
        return None, None, None
    
    # Converter memória de MB para GB
    memoria_gb = memoria_gb / 1024 if memoria_gb > 1024 else memoria_gb
    
    # Calcular diferença ponderada (priorizamos memória sobre CPU)
    df_precos['diferenca_cpu'] = abs(df_precos['vCPU(s)'] - cpu_cores)
    df_precos['diferenca_memoria'] = abs(df_precos['RAM_GB'] - memoria_gb)
    
    # Score ponderado (memória tem peso 2, CPU peso 1)
    df_precos['score'] = (df_precos['diferenca_cpu'] * 1) + (df_precos['diferenca_memoria'] * 2)
    
    # Encontrar a VM com menor score
    melhor_vm = df_precos.loc[df_precos['score'].idxmin()]
    
    return melhor_vm['Instance'], melhor_vm['Preco_por_hora_USD'], melhor_vm

def calcular_custo_azure(tempo_segundos, preco_por_hora_usd):
    """Calcula o custo total baseado no tempo de execução."""
    if tempo_segundos is None or tempo_segundos == float('inf') or preco_por_hora_usd is None:
        return None
    
    # Converter segundos para horas
    tempo_horas = tempo_segundos / 3600
    
    # Calcular custo total
    custo_total_usd = tempo_horas * preco_por_hora_usd
    
    return custo_total_usd

def obter_dados_azure_instance(azure_instance, df_precos):
    """Obtém dados de uma instância Azure específica."""
    if df_precos is None or azure_instance is None:
        return None, None, None
    
    # Procurar a instância na lista
    instancia_match = df_precos[df_precos['Instance'].str.lower() == azure_instance.lower()]
    
    if instancia_match.empty:
        print(f"⚠️ Instância '{azure_instance}' não encontrada na lista de preços.")
        print(f"\n📋 INSTÂNCIAS AZURE DISPONÍVEIS:")
        for idx, row in df_precos.iterrows():
            print(f"  - {row['Instance']}: {row['vCPU(s)']} vCPUs, {row['RAM']}, ${row['Preco_por_hora_USD']:.4f}/hora")
        return None, None, None
    
    dados_vm = instancia_match.iloc[0]
    return dados_vm['Instance'], dados_vm['Preco_por_hora_USD'], dados_vm

def listar_instancias_disponiveis():
    """Lista todas as instâncias Azure disponíveis."""
    df_precos = carregar_precos_azure()
    if df_precos is not None:
        print("\n📋 INSTÂNCIAS AZURE DISPONÍVEIS:")
        for idx, row in df_precos.iterrows():
            print(f"  - {row['Instance']}: {row['vCPU(s)']} vCPUs, {row['RAM']}, ${row['Preco_por_hora_USD']:.4f}/hora")
        return df_precos['Instance'].tolist()
    return []

def recomendar_biblioteca_simples(dataset_path, tem_joins=False, tem_groupby=False):
    """Versão simplificada que usa apenas configuração do sistema atual."""
    return recomendar_biblioteca(dataset_path, tem_joins, tem_groupby)

def recomendar_biblioteca_customizada(dataset_path, vcpus, memoria_gb, tem_joins=False, tem_groupby=False):
    """Versão para configuração de hardware customizada."""
    return recomendar_biblioteca(dataset_path, tem_joins, tem_groupby, 
                               custom_vcpus=vcpus, custom_memoria_gb=memoria_gb)

def recomendar_biblioteca_azure(dataset_path, instance_name, tem_joins=False, tem_groupby=False):
    """Versão para usar instância Azure específica."""
    return recomendar_biblioteca(dataset_path, tem_joins, tem_groupby, 
                               azure_instance=instance_name)

def coletar_info_sistema():
    return {
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "frequencia_cpu_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "memoria_total_mb": psutil.virtual_memory().total / (1024 ** 2),
        "disco_total_gb": psutil.disk_usage('/').total / (1024 ** 3),
    }

def analisar_amostra(path, nrows=5000):
    ext = Path(path).suffix.lower().replace('.', '')
    if ext == 'csv':
        df = pd.read_csv(path, nrows=nrows)
    elif ext == 'parquet':
        df = pd.read_parquet(path)
    elif ext == 'json':
        df = pd.read_json(path, lines=True, nrows=nrows)
    else:
        raise ValueError("Formato de arquivo não suportado.")

    tipos = df.dtypes
    total = len(tipos)

    numericos = tipos.apply(lambda t: pd.api.types.is_numeric_dtype(t)).sum()
    strings = tipos.apply(lambda t: pd.api.types.is_string_dtype(t)).sum()
    datetimes = tipos.apply(lambda t: pd.api.types.is_datetime64_any_dtype(t)).sum()

    return {
        "num_linhas": len(df),
        "num_colunas": df.shape[1],
        "percentual_numerico": numericos / total,
        "percentual_string": strings / total,
        "percentual_datetime": datetimes / total,
    }

def aplicar_preprocessamento_sucesso(entrada_dados):
    """Aplica o mesmo pré-processamento usado no treinamento do modelo de sucesso."""
    dados = entrada_dados.copy()
    
    # Feature principal de proporção de memória (mesmo que no treinamento)
    if 'tamanho_dataset_nominal_mb' in dados and 'memoria_total_mb' in dados:
        dados['proporcao_memoria'] = dados['tamanho_dataset_nominal_mb'] / dados['memoria_total_mb']
    
    return dados

def predizer_sucesso(entrada_dados):
    """Prediz se a operação será bem-sucedida usando o modelo campeão de sucesso."""
    try:
        # Aplicar o mesmo pré-processamento usado no treinamento
        dados_processados = aplicar_preprocessamento_sucesso(entrada_dados)
        
        # Preparar dados para o modelo de sucesso
        df_entrada = pd.DataFrame([dados_processados])
        
        # Selecionar apenas as features que o modelo de sucesso conhece
        features_disponiveis = [col for col in success_features if col in df_entrada.columns]
        df_entrada_success = df_entrada[features_disponiveis].copy()
        
        # Garantir que as colunas tenham os nomes corretos
        df_entrada_success.columns = features_disponiveis
        
        # Predizer probabilidade de sucesso
        prob_sucesso = success_model.predict_proba(df_entrada_success)[0, 1]
        sera_sucesso = prob_sucesso >= success_threshold
        
        return sera_sucesso, prob_sucesso
    except Exception as e:
        print(f"⚠️ Erro na predição de sucesso: {e}")
        return True, 0.5  # Assume sucesso por padrão

def aplicar_preprocessamento_tempo(entrada_dados):
    """Aplica o mesmo pré-processamento usado no treinamento do modelo de tempo."""
    dados = entrada_dados.copy()
    
    # Criar features de interação (mesmo que no treinamento)
    if 'tamanho_dataset_nominal_mb' in dados and 'num_linhas' in dados:
        dados['tamanho_por_linha'] = dados['tamanho_dataset_nominal_mb'] / dados['num_linhas']
    
    if 'num_linhas' in dados and 'num_colunas' in dados:
        dados['linhas_por_coluna'] = dados['num_linhas'] / dados['num_colunas']
    
    # Criar features de complexidade
    dados['complexidade_operacao'] = dados.get('tem_joins', 0) + dados.get('tem_groupby', 0)
    
    # Normalizar percentuais para somarem 1
    percent_cols = ['percentual_numerico', 'percentual_string', 'percentual_datetime']
    if all(col in dados for col in percent_cols):
        total = dados['percentual_numerico'] + dados['percentual_string'] + dados['percentual_datetime']
        if total > 0:
            dados['percentual_numerico'] = dados['percentual_numerico'] / total
            dados['percentual_string'] = dados['percentual_string'] / total
            dados['percentual_datetime'] = dados['percentual_datetime'] / total
    
    # Log transform para variáveis numéricas (MESMO QUE NO TREINAMENTO)
    numeric_cols = [
        'tamanho_dataset_nominal_mb',
        'tamanho_dataset_bytes', 
        'num_linhas',
        'num_colunas',
        'leitura_bytes',
        'escrita_bytes'
    ]
    
    for col in numeric_cols:
        if col in dados:
            dados[f'{col}_log'] = np.log1p(dados[col])
    
    return dados

def estimar_tempo(entrada_dados):
    """Estima o tempo de execução usando o modelo campeão de tempo."""
    try:
        # Aplicar o mesmo pré-processamento usado no treinamento
        dados_processados = aplicar_preprocessamento_tempo(entrada_dados)
        
        # Preparar dados para o modelo de tempo
        df_entrada = pd.DataFrame([dados_processados])
        
        # Selecionar apenas as features que o modelo de tempo conhece
        features_disponiveis = [col for col in time_features if col in df_entrada.columns]
        df_entrada_time = df_entrada[features_disponiveis].copy()
        
        # Garantir que as colunas tenham os nomes corretos
        df_entrada_time.columns = features_disponiveis
        
        # Predizer tempo (lembrar que o modelo foi treinado com log transform)
        tempo_log = time_model.predict(df_entrada_time)[0]
        tempo_previsto = np.expm1(tempo_log)  # Reverter log transform
        
        return max(tempo_previsto, 0.1)  # Garantir tempo mínimo
    except Exception as e:
        print(f"⚠️ Erro na estimativa de tempo: {e}")
        return 999.0  # Tempo alto por padrão

def recomendar_biblioteca(dataset_path, tem_joins=False, tem_groupby=False, 
                         custom_vcpus=None, custom_memoria_gb=None, azure_instance=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Arquivo de dataset não encontrado.")

    # Validar parâmetros de entrada
    if (custom_vcpus is not None and custom_memoria_gb is None) or (custom_vcpus is None and custom_memoria_gb is not None):
        raise ValueError("Ambos custom_vcpus e custom_memoria_gb devem ser fornecidos juntos ou nenhum dos dois.")
    
    if azure_instance is not None and (custom_vcpus is not None or custom_memoria_gb is not None):
        raise ValueError("Não é possível especificar azure_instance junto com custom_vcpus/custom_memoria_gb.")

    # Carregar dados de preços da Azure primeiro (necessário para determinar configuração)
    df_precos_azure = carregar_precos_azure()
    
    # Determinar configuração de hardware a ser usada
    if custom_vcpus is not None and custom_memoria_gb is not None:
        # Usar configuração customizada
        vcpus_para_calculo = custom_vcpus
        memoria_mb_para_calculo = custom_memoria_gb * 1024  # Converter GB para MB
        fonte_config = "CUSTOMIZADA"
        print(f"\n🔧 CONFIGURAÇÃO CUSTOMIZADA:")
        print(f"  - vCPUs: {vcpus_para_calculo}")
        print(f"  - Memória: {custom_memoria_gb} GB")
    elif azure_instance is not None:
        # Usar especificações da instância Azure
        if df_precos_azure is not None:
            vm_temp, preco_temp, dados_vm_temp = obter_dados_azure_instance(azure_instance, df_precos_azure)
            if vm_temp is not None:
                vcpus_para_calculo = dados_vm_temp['vCPU(s)']
                memoria_mb_para_calculo = dados_vm_temp['RAM_GB'] * 1024  # Converter GB para MB
                fonte_config = "INSTÂNCIA AZURE"
                print(f"\n☁️ CONFIGURAÇÃO DA INSTÂNCIA AZURE:")
                print(f"  - Instância: {azure_instance}")
                print(f"  - vCPUs: {vcpus_para_calculo}")
                print(f"  - Memória: {dados_vm_temp['RAM_GB']} GB")
            else:
                # Fallback para sistema atual se instância não for encontrada
                info_sistema = coletar_info_sistema()
                vcpus_para_calculo = info_sistema['nucleos_logicos']
                memoria_mb_para_calculo = info_sistema['memoria_total_mb']
                fonte_config = "SISTEMA ATUAL (FALLBACK)"
                print(f"\n🖥️ USANDO SISTEMA ATUAL (INSTÂNCIA AZURE INVÁLIDA):")
                for k, v in info_sistema.items():
                    print(f"  - {k.replace('_', ' ').capitalize()}: {round(v, 2) if isinstance(v, float) else v}")
        else:
            # Fallback para sistema atual se não conseguir carregar preços
            info_sistema = coletar_info_sistema()
            vcpus_para_calculo = info_sistema['nucleos_logicos']
            memoria_mb_para_calculo = info_sistema['memoria_total_mb']
            fonte_config = "SISTEMA ATUAL (FALLBACK)"
            print(f"\n🖥️ USANDO SISTEMA ATUAL (ERRO AO CARREGAR PREÇOS AZURE):")
            for k, v in info_sistema.items():
                print(f"  - {k.replace('_', ' ').capitalize()}: {round(v, 2) if isinstance(v, float) else v}")
    else:
        # Usar configuração do sistema atual
        info_sistema = coletar_info_sistema()
        vcpus_para_calculo = info_sistema['nucleos_logicos']
        memoria_mb_para_calculo = info_sistema['memoria_total_mb']
        fonte_config = "SISTEMA ATUAL"
        print(f"\n🖥️ CONFIGURAÇÃO DO SISTEMA ATUAL:")
        for k, v in info_sistema.items():
            print(f"  - {k.replace('_', ' ').capitalize()}: {round(v, 2) if isinstance(v, float) else v}")

    info_dados = analisar_amostra(dataset_path)

    print("\nPerfil do dataset analisado:")
    print(f"  - Arquivo: {dataset_path}")
    for k, v in info_dados.items():
        print(f"  - {k.replace('_', ' ').capitalize()}: {round(v, 4) if isinstance(v, float) else v}")

    tamanho_dataset_bytes = os.path.getsize(dataset_path)
    tamanho_dataset_nominal_mb = tamanho_dataset_bytes / (1024 * 1024)

    # Determinar VM para cálculo de custos
    vm_equivalente = None
    preco_por_hora = None
    dados_vm = None
    
    if df_precos_azure is not None:
        if azure_instance is not None:
            # Usar instância Azure específica
            vm_equivalente, preco_por_hora, dados_vm = obter_dados_azure_instance(azure_instance, df_precos_azure)
            fonte_vm = "INSTÂNCIA ESPECÍFICA"
        else:
            # Encontrar VM mais próxima da configuração
            vm_equivalente, preco_por_hora, dados_vm = encontrar_vm_mais_proxima(
                vcpus_para_calculo, 
                memoria_mb_para_calculo,
                df_precos_azure
            )
            fonte_vm = "MATCHING AUTOMÁTICO"
        
        if vm_equivalente:
            print(f"\n💰 VM AZURE PARA CÁLCULO DE CUSTOS ({fonte_vm}):")
            print(f"  - Instância: {vm_equivalente}")
            print(f"  - vCPUs: {dados_vm['vCPU(s)']}")
            print(f"  - RAM: {dados_vm['RAM']}")
            print(f"  - Custo por hora: ${preco_por_hora:.4f} USD")
            print(f"  - Configuração base: {fonte_config}")

    resultados = {}
    analises_sucesso = {}
    custos_azure = {}
    
    print(f"\n🔍 ANÁLISE COM MODELOS CAMPEÕES:")
    
    # Preparar informações do sistema para os modelos
    if fonte_config == "CUSTOMIZADA":
        # Criar um dicionário com configuração customizada
        info_sistema_para_modelo = {
            "nucleos_fisicos": custom_vcpus // 2,  # Estimativa: metade dos vCPUs são físicos
            "nucleos_logicos": custom_vcpus,
            "frequencia_cpu_max": 3000.0,  # Valor padrão estimado
            "memoria_total_mb": custom_memoria_gb * 1024,
            "disco_total_gb": 1000.0,  # Valor padrão estimado
        }
    elif fonte_config == "INSTÂNCIA AZURE":
        # Usar especificações da instância Azure
        info_sistema_para_modelo = {
            "nucleos_fisicos": vcpus_para_calculo // 2,  # Estimativa: metade dos vCPUs são físicos
            "nucleos_logicos": vcpus_para_calculo,
            "frequencia_cpu_max": 3000.0,  # Valor padrão estimado
            "memoria_total_mb": memoria_mb_para_calculo,
            "disco_total_gb": 1000.0,  # Valor padrão estimado
        }
    else:
        # Usar informações reais do sistema
        info_sistema_para_modelo = info_sistema

    for biblioteca in ["pandas", "polars", "spark", "duckdb"]:
        entrada = {
            "biblioteca": biblioteca,
            "dataset_formato": Path(dataset_path).suffix.replace('.', ''),
            "tamanho_dataset_nominal_mb": tamanho_dataset_nominal_mb,
            "cpu_medio_execucao": 50.0,
            "memoria_media_execucao": 1000.0,
            "leitura_bytes": tamanho_dataset_bytes,
            "escrita_bytes": 0,
            "tamanho_dataset_bytes": tamanho_dataset_bytes,
            "tem_joins": tem_joins,
            "tem_groupby": tem_groupby,
            **info_sistema_para_modelo,
            **info_dados
        }

        # 1. Predizer se será bem-sucedida
        sera_sucesso, prob_sucesso = predizer_sucesso(entrada)
        analises_sucesso[biblioteca] = (sera_sucesso, prob_sucesso)
        
        # 2. Estimar tempo (apenas se previsto como sucesso)
        if sera_sucesso:
            tempo_previsto = estimar_tempo(entrada)
            resultados[biblioteca] = tempo_previsto
            
            # 3. Calcular custo na Azure (se temos dados de preços)
            if preco_por_hora is not None:
                custo_azure = calcular_custo_azure(tempo_previsto, preco_por_hora)
                custos_azure[biblioteca] = custo_azure
            else:
                custos_azure[biblioteca] = None
        else:
            resultados[biblioteca] = float('inf')  # Tempo infinito para falhas
            custos_azure[biblioteca] = None

    # Exibir resultados
    print(f"\n📊 PREDIÇÕES DE SUCESSO:")
    for lib, (sucesso, prob) in analises_sucesso.items():
        status = "✅ SUCESSO" if sucesso else "❌ FALHA"
        print(f"  - {lib.upper()}: {status} (probabilidade: {prob:.1%})")

    print(f"\n⏱️ ESTIMATIVAS DE TEMPO:")
    bibliotecas_viáveis = []
    for lib, tempo in resultados.items():
        if tempo != float('inf'):
            custo_info = ""
            if custos_azure[lib] is not None:
                custo_info = f" (Custo Azure: ${custos_azure[lib]:.6f} USD)"
            print(f"  - {lib.upper()}: {tempo:.2f} segundos{custo_info}")
            bibliotecas_viáveis.append((lib, tempo))
        else:
            print(f"  - {lib.upper()}: FALHA PREVISTA")

    if bibliotecas_viáveis:
        melhor_biblioteca = min(bibliotecas_viáveis, key=lambda x: x[1])[0]
        print(f"\n🏆 BIBLIOTECA RECOMENDADA: {melhor_biblioteca.upper()}")
        
        # Mostrar probabilidade de sucesso da biblioteca recomendada
        prob_sucesso = analises_sucesso[melhor_biblioteca][1]
        print(f"   • Probabilidade de sucesso: {prob_sucesso:.1%}")
        print(f"   • Tempo estimado: {resultados[melhor_biblioteca]:.2f} segundos")
        
        # Mostrar custo na Azure se disponível
        if custos_azure[melhor_biblioteca] is not None and vm_equivalente:
            print(f"   • Custo estimado na Azure: ${custos_azure[melhor_biblioteca]:.6f} USD")
            print(f"   • VM Utilizada: {vm_equivalente}")
        
        return melhor_biblioteca
    else:
        print(f"\n⚠️ NENHUMA BIBLIOTECA RECOMENDADA")
        print(f"   • Todas as bibliotecas têm alta probabilidade de falha")
        print(f"   • Considere usar um dataset menor ou mais memória")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    dataset_path = "C:/Users/lucas/PycharmProjects/Lucas-TCC_2-2025_01/app/datasets/example.csv"
    
    # Opção 1: Usar configuração do sistema atual (comportamento padrão)
    print("=== USANDO CONFIGURAÇÃO DO SISTEMA ATUAL ===")
    resultado = recomendar_biblioteca(dataset_path, tem_joins=False, tem_groupby=False)
    
    # Opção 2: Usar configuração customizada de hardware
    print("\n\n=== USANDO CONFIGURAÇÃO CUSTOMIZADA ===")
    resultado = recomendar_biblioteca(dataset_path, tem_joins=False, tem_groupby=False, 
                                     custom_vcpus=16, custom_memoria_gb=64)
    
    # Opção 3: Usar instância Azure específica
    print("\n\n=== USANDO INSTÂNCIA AZURE ESPECÍFICA ===")
    resultado = recomendar_biblioteca(dataset_path, tem_joins=False, tem_groupby=False, 
                                     azure_instance="D32as v6")

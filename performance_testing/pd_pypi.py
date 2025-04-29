import pandas as pd
import argparse
import os

def processar_pasta_pypi(path: str):
    all_top1000 = []
    for filename in os.listdir(path):
        if filename.endswith(".csv") or filename.endswith(".json") or filename.endswith(".parquet"):
            file_path = os.path.join(path, filename)

            # ========================
            # 1. Leitura do arquivo
            # ========================
            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(".json"):
                # Ajuste para lines=True se o arquivo JSON estiver nesse formato
                df = pd.read_json(file_path, lines=True)
            else:
                df = pd.read_csv(file_path)

            # ========================
            # 2. Limpeza básica
            # ========================
            # Remove valores nulos
            df.dropna(inplace=True)
            # Remove duplicados
            df.drop_duplicates(inplace=True)

            # ========================
            # 3. Conversão de data/hora
            # ========================
            # Supondo que existe uma coluna 'upload_time' (ou 'date', 'timestamp', etc.)
            # Ajuste o nome da coluna para o que existir no seu dataset
            if "upload_time" in df.columns:
                df["upload_time"] = pd.to_datetime(df["upload_time"], errors="coerce")
            else:
                raise ValueError("Não foi encontrada a coluna 'upload_time' para extrair o mês. Ajuste o nome da coluna de data/hora no script.")

            # ========================
            # 4. Criação da coluna de mês
            # ========================
            # Se quiser também considerar o ano, use duas colunas, ex.: year e month
            df["month"] = df["upload_time"].dt.month
            df["year"] = df["upload_time"].dt.year

            # ========================
            # 5. Filtro de downloads
            # ========================
            # Certifique-se de ter a coluna 'downloads' no seu dataset
            if "downloads" not in df.columns:
                raise ValueError("Coluna 'downloads' não encontrada no DataFrame. Ajuste o script para as colunas adequadas.")

            # Exemplo de filtro para downloads > 0
            df = df[df["downloads"] > 0]

            # ========================
            # 6. Agrupamento por mês e pacotes
            # ========================
            # Ajuste o nome da coluna de pacotes se for diferente de "package_name"
            if "package_name" not in df.columns:
                raise ValueError("Coluna 'package_name' não encontrada. Ajuste de acordo com seu dataset.")

            # Calcula a soma de downloads por ano, mês e pacote
            df_grouped = (
                df.groupby(["year", "month", "package_name"], as_index=False)["downloads"]
                .sum()
            )

            # ========================
            # 7. Ranking dos 1000 pacotes mais baixados por mês
            # ========================
            # Para cada combinação de (ano, mês), calcular o rank de cada pacote com base em downloads
            df_grouped["rank"] = df_grouped.groupby(["year", "month"])["downloads"] \
                                           .rank(method="dense", ascending=False)

            # Filtro para manter apenas os top 1000 em cada mês
            df_top1000 = df_grouped[df_grouped["rank"] <= 1000].copy()

            all_top1000.append(df_top1000)

    if all_top1000:
        df_final = pd.concat(all_top1000, ignore_index=True)

        output_folder = "processed_data/"
        os.makedirs(output_folder, exist_ok=True)

        # Salva em Parquet o ranking final
        df_final.to_parquet(os.path.join(output_folder, "pypi_top1000_monthly.parquet"), index=False)

        # Estatísticas descritivas do ranking final
        descriptive_stats = df_final.describe()
        descriptive_stats.to_csv(os.path.join(output_folder, "descriptive_stats_top1000_pypi.csv"))
    else:
        print("Nenhum arquivo válido encontrado na pasta especificada.")

def main():
    parser = argparse.ArgumentParser(description="Script de processamento de dados PyPI - Ranking Top 1000 pacotes por mês")
    parser.add_argument('--input', type=str, required=True, help='Caminho para a pasta de entrada contendo arquivos CSV, JSON ou Parquet')
    args = parser.parse_args()

    processar_pasta_pypi(args.input)

if __name__ == "__main__":
    main()
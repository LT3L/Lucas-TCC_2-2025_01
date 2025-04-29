import os
import pandas as pd
from google.cloud import bigquery

def fetch_and_save_pypi_data_in_pages(
    output_folder="pypi_pages",
    page_size=50_000_000
):
    client = bigquery.Client()

    query = """
    SELECT 
      details.python AS python_version, 
      file.project AS project_name,
      country_code,
      DATE_TRUNC(DATE(timestamp), DAY) AS download_date
    FROM `bigquery-public-data.pypi.file_downloads`
    WHERE DATE(timestamp) BETWEEN '2025-01-01' AND '2025-01-02'
    """

    print("Enviando consulta para o BigQuery...")
    job = client.query(query)

    # 'job.result(...)' retorna um RowIterator, que pode ser paginado
    rows_iter = job.result(page_size=page_size)

    print("Consulta executada. Iniciando download paginado...")

    # Garante que a pasta de saída exista
    os.makedirs(output_folder, exist_ok=True)

    total_rows = 0
    page_index = 0

    # Iteramos sobre cada página de resultados
    for page in rows_iter.pages:
        # Convertemos as linhas da página em uma lista de dict
        page_rows = list(page)  # cada elemento é um 'Row'
        df_page = pd.DataFrame([dict(row) for row in page_rows])

        # O número de linhas nesta página
        num_rows_page = len(df_page)
        total_rows += num_rows_page

        # Monta o nome do arquivo de saída (ex: pypi_part_00000.parquet)
        page_filename = f"pypi_part_{page_index:05d}.parquet"
        page_path = os.path.join(output_folder, page_filename)

        # Salva em Parquet
        df_page.to_parquet(page_path, index=False)

        print(f"Página {page_index} salva com {num_rows_page} linhas. Total acumulado: {total_rows}. Arquivo: {page_path}")

        page_index += 1

    print(f"Download paginado concluído. Total de páginas: {page_index}. Total de linhas: {total_rows}")

if __name__ == "__main__":
    fetch_and_save_pypi_data_in_pages(
        output_folder="pypi_pages",
        page_size=50_000_000
    )
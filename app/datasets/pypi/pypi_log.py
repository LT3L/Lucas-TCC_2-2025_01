import os
import requests
import gzip
import shutil

def download_single_pypi_log(year, month, day, hour, output_dir="pypi_logs"):
    """
    Baixa e descompacta um único log horário do PyPI da storage pública do GCP.
    Ex: https://storage.googleapis.com/pypi-downloads/logs/2024/01/01/00.json.gz
    """
    url = f"https://storage.googleapis.com/pypi-downloads/logs/{year:04d}/{month:02d}/{day:02d}/{hour:02d}.json.gz"
    os.makedirs(output_dir, exist_ok=True)

    local_gz_path = os.path.join(output_dir, f"{year:04d}-{month:02d}-{day:02d}_{hour:02d}.json.gz")
    local_json_path = local_gz_path[:-3]  # remove .gz

    try:
        print(f"🔽 Baixando: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(local_gz_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        print("🗜️  Descompactando...")
        with gzip.open(local_gz_path, "rb") as f_in, open(local_json_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        print(f"✅ Arquivo salvo em: {local_json_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao baixar: {e}")

# Exemplo: baixar 1 de janeiro de 2024, 00h
if __name__ == "__main__":
    download_single_pypi_log(2024, 1, 1, 0)
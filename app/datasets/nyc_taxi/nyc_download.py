import os
import requests
import shutil


def download_nyc_taxi_month(year, month, output_dir="raw"):
    """
    Baixa e salva um arquivo mensal do NYC Yellow Taxi Trip Data.
    Exemplo: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"

    local_path = os.path.join(output_dir, filename)

    if os.path.exists(local_path):
        print(f"⏩ Arquivo já existe, pulando: {local_path}")
        return

    print(f"🔽 Baixando: {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        print(f"✅ Arquivo salvo em: {local_path}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao baixar o arquivo: {e}")


def download_until_10gb(start_year=2025, start_month=3, output_dir="raw"):
    import datetime

    total_size = 0
    target_bytes = 10 * 1024 ** 3  # 10 GB
    current = datetime.datetime(start_year, start_month, 1)

    while total_size < target_bytes:
        year = current.year
        month = current.month

        download_nyc_taxi_month(year, month, output_dir)

        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if f.endswith(".parquet")
        )
        print(f"📦 Tamanho total atual: {total_size / (1024 ** 3):.2f} GB")

        # Volta um mês
        if month == 1:
            current = datetime.datetime(year - 1, 12, 1)
        else:
            current = datetime.datetime(year, month - 1, 1)


# Exemplo: baixa arquivos do NYC Taxi até atingir 10 GB
if __name__ == "__main__":
    download_until_10gb()

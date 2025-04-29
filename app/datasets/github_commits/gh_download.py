import os
import requests
import gzip
import shutil

def download_gharchive_hour(year, month, day, hour, output_dir="raw"):
    """
    Baixa e descompacta um arquivo de evento horário do GH Archive.
    Ex: https://data.gharchive.org/2024-01-01-0.json.gz
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{year:04d}-{month:02d}-{day:02d}-{hour}.json.gz"
    url = f"https://data.gharchive.org/{filename}"

    local_gz_path = os.path.join(output_dir, filename)
    local_json_path = local_gz_path[:-3]  # remove .gz

    if os.path.exists(local_json_path):
        print(f"⏩ Arquivo já existe, pulando: {local_json_path}")
        return

    print(f"🔽 Baixando: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(local_gz_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        print("🗜️  Descompactando...")
        with gzip.open(local_gz_path, "rb") as f_in, open(local_json_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        print(f"✅ Arquivo salvo em: {local_json_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao baixar ou descompactar: {e}")

def download_until_50gb(start_year=2024, start_month=1, start_day=1, output_dir="gharchive_logs"):
    import datetime

    total_size = 0
    target_bytes = 50 * 1024 ** 3  # 50 GB
    current = datetime.datetime(start_year, start_month, start_day, 0)

    while total_size < target_bytes:
        year = current.year
        month = current.month
        day = current.day
        hour = current.hour

        download_gharchive_hour(year, month, day, hour, output_dir)

        # Atualiza tamanho total descompactado
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        )
        print(f"📦 Tamanho total atual: {total_size / (1024 ** 3):.2f} GB")

        # Avança uma hora
        current += datetime.timedelta(hours=1)

# Exemplo: baixa logs até atingir 50 GB
if __name__ == "__main__":
    download_until_50gb()
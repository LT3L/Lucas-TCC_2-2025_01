import os
import pandas as pd
import json
import datetime

def log_execution(entry, LOG_FILE='log_execucoes.jsonl'):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def export_sample(subset, fmt, path):
    if fmt == 'parquet':
        subset.to_parquet(path, index=False)
    elif fmt == 'csv':
        subset.to_csv(path, index=False)
    elif fmt == 'json':
        subset.to_json(path, orient='records', lines=True)

def find_row_count_for_target_size(df, fmt, target_mb, TOLERANCE=0.05):
    low, high = 100, len(df)
    best_rows = None

    temp_path = f'temp_{fmt}.{fmt}'

    while low <= high:
        mid = (low + high) // 2
        subset = df.iloc[:mid]
        export_sample(subset, fmt, temp_path)
        size = get_file_size_mb(temp_path)

        if abs(size - target_mb) / target_mb <= TOLERANCE:
            best_rows = mid
            break
        elif size < target_mb:
            low = mid + 1
        else:
            high = mid - 1

    if best_rows:
        os.remove(temp_path)
    return best_rows


def estimate_rows_for_size(df, fmt, target_mb, sample_size=1000):
    sample = df.iloc[:min(sample_size, len(df))]
    temp_path = f'temp_sample_{fmt}.{fmt}'
    export_sample(sample, fmt, temp_path)
    sample_size_mb = get_file_size_mb(temp_path)
    os.remove(temp_path)

    avg_row_size_mb = sample_size_mb / len(sample)
    estimated_rows = int(target_mb / avg_row_size_mb)
    return estimated_rows

def find_row_count_for_target_size_fast(df, fmt, target_mb, tolerance=0.05):
    estimated_rows = estimate_rows_for_size(df, fmt, target_mb)

    for attempt in range(3):  # tenta no máximo 3 ajustes
        subset = df.iloc[:estimated_rows]
        temp_path = f'temp_{fmt}.{fmt}'
        export_sample(subset, fmt, temp_path)
        size_mb = get_file_size_mb(temp_path)

        diff_ratio = abs(size_mb - target_mb) / target_mb
        if diff_ratio <= tolerance:
            os.remove(temp_path)
            return estimated_rows

        # Ajuste simples: aumenta ou reduz linhas proporcionalmente
        if size_mb < target_mb:
            estimated_rows = int(estimated_rows * 1.1)
        else:
            estimated_rows = int(estimated_rows * 0.9)

        os.remove(temp_path)

    return estimated_rows  # mesmo que esteja levemente fora

def generate_files():
    TARGET_SIZES_MB = [10, 100, 1000]  # tamanhos desejados
    FORMATS = ['parquet', 'csv', 'json']
    INPUT_FOLDER = 'NYC_raw_download'
    OUTPUT_FOLDER = 'NYC_sized'
    TOLERANCE = 0.05  # 5% de tolerância no tamanho do arquivo

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 🔄 Carrega todos os Parquets
    print("🔄 Lendo arquivos...")
    df = pd.concat([pd.read_parquet(os.path.join(INPUT_FOLDER, f))
                    for f in os.listdir(INPUT_FOLDER)
                    if f.endswith('.parquet')])

    SEED = 42  # escolha qualquer número fixo
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    LOG_FILE = 'log_execucoes.jsonl'
# 📦 Geração dos arquivos finais
    for fmt in FORMATS:
        print(f"\n📂 Format: {fmt}")
        for size_mb in TARGET_SIZES_MB:
            print(f"🎯 Gerando {size_mb}MB...")

            if fmt == "parquet":
                n_rows = find_row_count_for_target_size(df, fmt, size_mb, TOLERANCE)

            else:
                n_rows = find_row_count_for_target_size_fast(df, fmt, size_mb, TOLERANCE)

            if not n_rows:
                print(f"❌ Não foi possível gerar {size_mb}MB em {fmt}")
                continue

            output_path = f"{OUTPUT_FOLDER}/{fmt}/amostra_{size_mb}MB.{fmt}"
            export_sample(df.iloc[:n_rows], fmt, output_path)
            real_size = get_file_size_mb(output_path)

            log_execution({
                "seed": SEED,
                "target_size_mb": size_mb,
                "format": fmt,
                "output_file": output_path,
                "n_rows": n_rows,
                "final_size_mb": round(real_size, 2),
                "timestamp" : str(datetime.datetime.now())
            })

            print(f"✅ {output_path} com {n_rows} linhas e {real_size:.2f}MB")



if __name__ == "__main__":
    generate_files()
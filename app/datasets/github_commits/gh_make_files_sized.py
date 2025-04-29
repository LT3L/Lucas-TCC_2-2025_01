import os
import pandas as pd
import json
import datetime
import duckdb

def log_execution(entry, LOG_FILE='log_execucoes.jsonl'):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def export_sample(subset, fmt, path):
    df = subset.copy()

    # Padroniza datetime para string no formato ISO 8601 (sem milissegundos)
    for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')

    if fmt == 'parquet':
        df.to_parquet(path, index=False)
    elif fmt == 'csv':
        df.to_csv(path, index=False)
    elif fmt == 'json':
        df.to_json(path, orient='records', lines=True)

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
    TARGET_SIZES_MB = [10, 100, 1000, 10_000, 50_000]
    FORMATS = ['csv', 'json']
    INPUT_FOLDER = '/Users/lucas.lima/Documents/Projects/TCC_2/app/datasets/github_commits/parquet_parts'
    OUTPUT_FOLDER = '/Users/lucas.lima/Documents/Projects/TCC_2/app/datasets/github_commits'
    TOLERANCE = 0.05

    remaining_needed = {fmt: {size: True for size in TARGET_SIZES_MB} for fmt in FORMATS}
    partial_buffers = {fmt: {size: pd.DataFrame() for size in TARGET_SIZES_MB} for fmt in FORMATS}
    part_counter = {fmt: {size: 0 for size in TARGET_SIZES_MB} for fmt in FORMATS}
    current_sizes = {fmt: {size: 0 for size in TARGET_SIZES_MB} for fmt in FORMATS}

    for file in sorted(os.listdir(INPUT_FOLDER)):
        if not file.endswith(".parquet"):
            continue

        df = pd.read_parquet(os.path.join(INPUT_FOLDER, file))

        for fmt in FORMATS:
            for size in TARGET_SIZES_MB:
                if not remaining_needed[fmt][size]:
                    continue

                if fmt == "parquet":
                    row_count = find_row_count_for_target_size(df, fmt, size, TOLERANCE)
                else:
                    row_count = find_row_count_for_target_size_fast(df, fmt, size, TOLERANCE)

                if row_count and row_count <= len(df):
                    out_dir = os.path.join(OUTPUT_FOLDER, f"{fmt}/{size}MB")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"amostra_{size}MB.{fmt}")
                    export_sample(df.iloc[:row_count], fmt, out_path)
                    final_size = get_file_size_mb(out_path)

                    log_execution({
                        "source_file": file,
                        "target_size_mb": size,
                        "format": fmt,
                        "output_file": out_path,
                        "n_rows": row_count,
                        "final_size_mb": round(final_size, 2),
                        "timestamp": str(datetime.datetime.now())
                    })

                    print(f"✅ {out_path} com {row_count} linhas e {final_size:.2f}MB")
                    remaining_needed[fmt][size] = False
                    partial_buffers[fmt][size] = pd.DataFrame()
                else:
                    out_dir = os.path.join(OUTPUT_FOLDER, f"{fmt}/{size}MB")
                    os.makedirs(out_dir, exist_ok=True)
                    part_n = part_counter[fmt][size]
                    out_path = os.path.join(out_dir, f"part_{part_n}_amostra_{size}MB.{fmt}")
                    export_sample(df, fmt, out_path)
                    part_counter[fmt][size] += 1

                    part_size = get_file_size_mb(out_path)
                    current_sizes[fmt][size] += part_size

                    log_execution({
                        "source_file": file,
                        "target_size_mb": size,
                        "format": fmt,
                        "output_file": out_path,
                        "n_rows": len(df),
                        "final_size_mb": round(part_size, 2),
                        "timestamp": str(datetime.datetime.now())
                    })

                    print(f"🧩 {out_path} parcial salvo com {len(df)} linhas e {part_size:.2f}MB")

                    if current_sizes[fmt][size] >= size * (1 - TOLERANCE):
                        remaining_needed[fmt][size] = False

if __name__ == "__main__":
    generate_files()
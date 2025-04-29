import os
import pandas as pd
from glob import glob

RAW_DIR = "raw"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parquet_parts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob(os.path.join(RAW_DIR, "*.json")))

for idx, file in enumerate(files):
    try:
        df_raw = pd.read_json(file, lines=True)
        df_norm = pd.json_normalize(df_raw.to_dict(orient="records"))

        out_path = os.path.join(OUTPUT_DIR, f"part_{idx:04d}.parquet")
        df_norm.to_parquet(out_path, index=False)
        print(f"✅ Salvo: {out_path}")
    except Exception as e:
        print(f"❌ Erro em {file}: {e}")

print(f"🎉 Total de arquivos salvos: {len(files)}")
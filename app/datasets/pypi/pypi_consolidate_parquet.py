import os
import pandas as pd
from glob import glob
import shutil
import math

# Configuration
INPUT_DIR = "pypi_export"
FORMATS = ["csv", "json", "parquet"]
SIZES_MB = [10, 100, 1000, 10000]
SIZE_MARGIN = 0.05  # 5% margin
CHUNK_SIZE = 10000  # Number of rows to process at once

# Create output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRS = {
    format: os.path.join(BASE_DIR, f"{format}") 
    for format in FORMATS
}

def setup_output_dirs():
    """Create the directory structure for all formats and sizes"""
    for format_type in FORMATS:
        # Create main format directory
        os.makedirs(OUTPUT_DIRS[format_type], exist_ok=True)
        # Create size-specific directories
        for size in SIZES_MB:
            size_dir = os.path.join(OUTPUT_DIRS[format_type], f"{size}MB")
            os.makedirs(size_dir, exist_ok=True)

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def get_dir_size_mb(dir_path):
    """Get total size of all files in directory in MB"""
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def copy_parquet_files(target_size_mb, input_files, output_dir):
    """Copy parquet files until reaching target size"""
    total_size = 0
    part_number = 1
    
    for file in input_files:
        file_size = get_file_size_mb(file)
        if total_size + file_size <= target_size_mb * (1 + SIZE_MARGIN):
            dest = os.path.join(output_dir, f"part_{part_number:04d}.parquet")
            shutil.copy2(file, dest)
            total_size += file_size
            part_number += 1
            print(f"✅ Copied Parquet file: {file}")
        else:
            break
    
    return total_size

def process_file_in_parts(file_path, target_size_mb, format_type, output_dir, part_number):
    """Process a single file and write as a numbered part"""
    total_rows = 0
    total_size = 0
    output_file = os.path.join(output_dir, f"part_{part_number:04d}.{format_type}")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    total_rows = len(df)
    
    # Calculate how many rows we need to write
    rows_to_write = min(total_rows, int(target_size_mb * 1e6 / (df.memory_usage(deep=True).sum() / total_rows)))
    
    if rows_to_write > 0:
        # Write the data
        if format_type == "csv":
            df.head(rows_to_write).to_csv(output_file, index=False)
        else:  # json
            df.head(rows_to_write).to_json(output_file, orient="records", lines=True)
        
        total_size = get_file_size_mb(output_file)
    
    return total_size

def adjust_last_part(target_size_mb, format_type, output_dir):
    """Adjust the last part to meet size requirements"""
    parts = sorted(glob(os.path.join(output_dir, f"part_*.{format_type}")))
    if not parts:
        return
    
    total_size = get_dir_size_mb(output_dir)
    if total_size > target_size_mb * (1 + SIZE_MARGIN):
        # Get the last part
        last_part = parts[-1]
        df = pd.read_csv(last_part) if format_type == "csv" else pd.read_json(last_part, lines=True)
        
        # Calculate how much we need to reduce
        excess_size = total_size - target_size_mb
        last_part_size = get_file_size_mb(last_part)
        
        if excess_size >= last_part_size:
            # If we need to remove more than the last part, remove it entirely
            os.remove(last_part)
            print(f"⚠️ Removed last part as it was too large")
        else:
            # Calculate the target size for the last part
            target_last_part_size = last_part_size - excess_size
            # Calculate how many rows to keep (approximate)
            rows_to_keep = int(len(df) * (target_last_part_size / last_part_size))
            
            if rows_to_keep > 0:
                # Keep only the calculated number of rows
                df = df.head(rows_to_keep)
                
                # Save the reduced part
                if format_type == "csv":
                    df.to_csv(last_part, index=False)
                else:  # json
                    df.to_json(last_part, orient="records", lines=True)
                print(f"✅ Adjusted last part to {rows_to_keep} rows")
            else:
                # If we can't keep any rows, remove the part
                os.remove(last_part)
                print(f"⚠️ Removed last part as it was too large")

# Setup directory structure
setup_output_dirs()

# Get all input files
files = sorted(glob(os.path.join(INPUT_DIR, "*.parquet")))
if not files:
    print(f"❌ No Parquet files found in {INPUT_DIR}")
    exit(1)

print(f"📊 Found {len(files)} Parquet files")

# Process each target size
for size_mb in SIZES_MB:
    print(f"\n🔄 Creating {size_mb}MB datasets...")
    
    # Handle Parquet format (copy original files)
    parquet_dir = os.path.join(OUTPUT_DIRS["parquet"], f"{size_mb}MB")
    if not os.path.exists(parquet_dir) or not os.listdir(parquet_dir):
        total_size = copy_parquet_files(size_mb, files, parquet_dir)
        print(f"✅ Total Parquet size: {total_size:.2f}MB")
    
    # Handle CSV and JSON formats (process in parts)
    for format_type in ["csv", "json"]:
        output_dir = os.path.join(OUTPUT_DIRS[format_type], f"{size_mb}MB")
        
        # Skip if directory already has files
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"✅ {format_type.upper()} dataset for {size_mb}MB already exists")
            continue
            
        print(f"Processing {format_type.upper()} format...")
        current_size = 0
        part_number = 1
        
        for file in files:
            if current_size >= size_mb * (1 + SIZE_MARGIN):
                break
                
            file_size = process_file_in_parts(file, size_mb, format_type, 
                                            output_dir, part_number)
            current_size += file_size
            part_number += 1
            
            print(f"✅ Created part {part_number-1:04d} from: {file}")
        
        # Adjust last part if needed
        adjust_last_part(size_mb, format_type, output_dir)
        
        # Print final size
        final_size = get_dir_size_mb(output_dir)
        print(f"✅ Total {format_type.upper()} size: {final_size:.2f}MB")



for size_mb in [50_000]:
    print(f"\n🔄 Creating {size_mb}MB datasets...")

    # Handle CSV and JSON formats (process in parts)
    for format_type in ["csv", "json"]:
        output_dir = os.path.join(OUTPUT_DIRS[format_type], f"{size_mb}MB")

        # Skip if directory already has files
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"✅ {format_type.upper()} dataset for {size_mb}MB already exists")
            continue

        print(f"Processing {format_type.upper()} format...")
        current_size = 0
        part_number = 1

        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if current_size >= size_mb * (1 + SIZE_MARGIN):
                break


            file_size = process_file_in_parts(file, size_mb, format_type,
                                              output_dir, part_number)
            current_size += file_size
            part_number += 1

            print(f"✅ Created part {part_number - 1:04d} from: {file}")

        # Adjust last part if needed
        adjust_last_part(size_mb, format_type, output_dir)

        # Print final size
        final_size = get_dir_size_mb(output_dir)
        print(f"✅ Total {format_type.upper()} size: {final_size:.2f}MB")


print("\n🎉 Dataset creation completed!")
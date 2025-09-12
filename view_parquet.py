#!/usr/bin/env python3
"""
Parquet data viewer script
Usage:
    python view_parquet.py --file data/train-00000-of-00013.parquet
    python view_parquet.py --file data/train-00000-of-00013.parquet --limit 10
"""

import argparse
import pandas as pd
import json
from pathlib import Path

def view_parquet(file_path, limit=None, show_columns=True, show_sample=True):
    """View parquet file contents"""
    try:
        # Load parquet file
        df = pd.read_parquet(file_path)
        
        print(f"📁 File: {file_path}")
        print(f"📊 Shape: {df.shape} (rows, columns)")
        print("="*60)
        
        if show_columns:
            print("📋 Columns:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col} ({df[col].dtype})")
            print("="*60)
        
        if show_sample:
            print("🔍 Sample data:")
            sample_df = df.head(limit) if limit else df.head(5)
            
            for idx, row in sample_df.iterrows():
                print(f"\n--- Record {idx+1} ---")
                for col in df.columns:
                    value = row[col]
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    elif isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False, indent=2)[:500] + "..."
                    print(f"{col}: {value}")
                print("-" * 40)
        
        # Statistics
        print(f"\n📈 Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing values:")
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count}")
        
    except Exception as e:
        print(f"❌ Error reading parquet file: {e}")

def main():
    parser = argparse.ArgumentParser(description="View parquet file contents")
    parser.add_argument("--file", required=True, help="Path to parquet file")
    parser.add_argument("--limit", type=int, help="Limit number of records to show")
    parser.add_argument("--no-columns", action="store_true", help="Don't show column info")
    parser.add_argument("--no-sample", action="store_true", help="Don't show sample data")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    view_parquet(
        file_path, 
        limit=args.limit,
        show_columns=not args.no_columns,
        show_sample=not args.no_sample
    )

if __name__ == "__main__":
    main()

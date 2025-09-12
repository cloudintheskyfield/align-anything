#!/usr/bin/env python3
"""
Debug script to check parquet image data format
"""

import pandas as pd
import base64
from pathlib import Path

def debug_image_data(file_path, limit=3):
    """Debug image data in parquet file"""
    df = pd.read_parquet(file_path)
    
    print(f"File: {file_path}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("="*60)
    
    # Check first few records
    for idx, row in df.head(limit).iterrows():
        print(f"\n--- Record {idx+1} ---")
        
        if 'image' in row:
            image_data = row['image']
            print(f"Image data type: {type(image_data)}")
            print(f"Image data length: {len(str(image_data))}")
            
            if isinstance(image_data, str):
                print(f"First 100 chars: {image_data[:100]}")
                print(f"Last 50 chars: {image_data[-50:]}")
                
                # Check if it's valid base64
                try:
                    decoded = base64.b64decode(image_data)
                    print(f"Base64 decode successful, size: {len(decoded)} bytes")
                    
                    # Check image format by magic bytes
                    if decoded.startswith(b'\xff\xd8\xff'):
                        print("Format: JPEG")
                    elif decoded.startswith(b'\x89PNG'):
                        print("Format: PNG")
                    elif decoded.startswith(b'GIF'):
                        print("Format: GIF")
                    elif decoded.startswith(b'RIFF') and b'WEBP' in decoded[:12]:
                        print("Format: WebP")
                    else:
                        print(f"Unknown format, first 16 bytes: {decoded[:16]}")
                        
                except Exception as e:
                    print(f"Base64 decode failed: {e}")
            else:
                print(f"Image data: {image_data}")
        
        print("-" * 40)

if __name__ == "__main__":
    file_path = "data/train-00000-of-00013.parquet"
    if Path(file_path).exists():
        debug_image_data(file_path)
    else:
        print(f"File not found: {file_path}")

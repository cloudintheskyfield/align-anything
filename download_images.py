#!/usr/bin/env python3
"""
Download test images for multimodal assessment
"""
import requests
import os
from pathlib import Path
import time

def download_image(url, filename, data_dir):
    """Download an image from URL and save to data directory"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        filepath = data_dir / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Test images with different scenarios
    images = [
        {
            "url": "https://picsum.photos/800/600?random=1",
            "filename": "test_image_1.jpg",
            "description": "Random landscape/nature image"
        },
        {
            "url": "https://picsum.photos/800/600?random=2", 
            "filename": "test_image_2.jpg",
            "description": "Random architectural image"
        },
        {
            "url": "https://picsum.photos/800/600?random=3",
            "filename": "test_image_3.jpg",
            "description": "Random urban scene"
        },
        {
            "url": "https://picsum.photos/800/600?random=4",
            "filename": "test_image_4.jpg", 
            "description": "Random portrait/people"
        },
        {
            "url": "https://picsum.photos/800/600?random=5",
            "filename": "test_image_5.jpg",
            "description": "Random abstract/artistic"
        }
    ]
    
    print("🖼️  开始下载测试图片...")
    
    successful_downloads = 0
    for i, img in enumerate(images, 1):
        print(f"\n📥 下载图片 {i}/5: {img['description']}")
        if download_image(img["url"], img["filename"], data_dir):
            successful_downloads += 1
        
        # Add delay between downloads to be respectful
        if i < len(images):
            time.sleep(1)
    
    print(f"\n🎉 下载完成! 成功下载 {successful_downloads}/{len(images)} 张图片")
    print(f"📁 图片保存在: {data_dir.absolute()}")
    
    # List downloaded files
    print("\n📋 已下载的文件:")
    for img_file in data_dir.glob("test_image_*.jpg"):
        size = img_file.stat().st_size / 1024  # KB
        print(f"   📸 {img_file.name} ({size:.1f} KB)")

if __name__ == "__main__":
    main()

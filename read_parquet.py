#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用于读取和查看 parquet 文件的脚本
"""

import pandas as pd
import os

def read_parquet_file(file_path):
    """
    读取 parquet 文件并显示基本信息
    
    Args:
        file_path (str): parquet 文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return None
        
        print(f"正在读取文件: {file_path}")
        print("=" * 50)
        
        # 读取 parquet 文件
        df = pd.read_parquet(file_path)
        
        # 显示基本信息
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n数据类型:")
        print(df.dtypes)
        
        print("\n前5行数据:")
        print(df.head())
        
        print("\n数据统计信息:")
        print(df.describe())
        
        # 检查缺失值
        print("\n缺失值统计:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        return df
        
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

def main():
    """主函数"""
    # parquet 文件路径
    parquet_file = "./data/train-00000-of-00013.parquet"
    
    print("Parquet 文件读取工具")
    print("=" * 50)
    
    # 读取并显示文件信息
    df = read_parquet_file(parquet_file)
    
    if df is not None:
        print(f"\n成功读取文件，共 {len(df)} 行数据")
        
        # 可选：保存为 CSV 文件以便查看
        save_csv = input("\n是否保存为 CSV 文件以便查看? (y/n): ").lower().strip()
        if save_csv == 'y':
            csv_file = "./data/train_sample.csv"
            # 只保存前1000行作为样本
            df.head(1000).to_csv(csv_file, index=False, encoding='utf-8')
            print(f"已保存前1000行数据到: {csv_file}")

if __name__ == "__main__":
    main()

"""调试Excel文件结构"""
import pandas as pd
from pathlib import Path

# 读取SCFI数据文件（不跳过任何行）
print("=== SCFI文件原始内容（前5行）===")
df_raw = pd.read_excel('data_source/上海出口集装箱运价指数.xlsx', header=None, nrows=5)
print(df_raw)
print(f"\n列名: {df_raw.columns.tolist()}")
print()

# 读取韩国出口数据文件（不跳过任何行）
print("=== 韩国出口文件原始内容（前5行）===")
df_korea_raw = pd.read_excel('data_source/韩国出口指数.xlsx', header=None, nrows=5)
print(df_korea_raw)
print(f"\n列名: {df_korea_raw.columns.tolist()}")
print()

# 测试各种读取方式
print("=== 测试1: header=None ===")
df1 = pd.read_excel('data_source/上海出口集装箱运价指数.xlsx', header=None)
print(f"形状: {df1.shape}")
print(f"列名: {df1.columns.tolist()}")
print(f"第一行: {df1.iloc[0].tolist()}")
print()

print("=== 测试2: header=0 ===")
df2 = pd.read_excel('data_source/上海出口集装箱运价指数.xlsx', header=0)
print(f"形状: {df2.shape}")
print(f"列名: {df2.columns.tolist()}")
print(f"第一行: {df2.iloc[0].tolist()}")
print()

print("=== 测试3: skiprows=1, header=0 ===")
df3 = pd.read_excel('data_source/上海出口集装箱运价指数.xlsx', skiprows=1, header=0)
print(f"形状: {df3.shape}")
print(f"列名: {df3.columns.tolist()}")
print(f"第一行: {df3.iloc[0].tolist()}")
print()

print("=== 测试4: skiprows=1 ===")
df4 = pd.read_excel('data_source/上海出口集装箱运价指数.xlsx', skiprows=1)
print(f"形状: {df4.shape}")
print(f"列名: {df4.columns.tolist()}")
print(f"第一行: {df4.iloc[0].tolist()}")

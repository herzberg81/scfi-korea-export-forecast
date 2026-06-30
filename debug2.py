"""调试Excel文件结构"""
import pandas as pd

# 测试韩国出口的读取方式
print("=== 韩国出口 skiprows=1, header=0 ===")
df2 = pd.read_excel('data_source/韩国出口指数.xlsx', skiprows=1, header=0)
print(f"形状: {df2.shape}")
print(f"列名: {df2.columns.tolist()}")
print(f"第一行: {df2.iloc[0].tolist()}")
print()
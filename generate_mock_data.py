"""
生成模拟数据脚本
用于测试SCFI海运运价预测系统
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(start_date='2016-01-01', num_weeks=500):
    """生成模拟的SCFI和韩国出口指数数据

    Args:
        start_date: 起始日期
        num_weeks: 生成周数（默认500周，约10年历史数据）
    """
    # 生成日期序列（每周）
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(weeks=i) for i in range(num_weeks)]

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 生成韩国出口先行指数（作为领先指标，会影响SCFI）
    # 先行指数呈现周期性波动，有一定的趋势
    korea_leading_base = np.array([100 + 0.05 * i + 15 * np.sin(i / 52 * 2 * np.pi) for i in range(num_weeks)])
    korea_leading = korea_leading_base + np.random.randn(num_weeks) * 5

    # 生成SCFI综合指数（与韩国出口先行指数有较强相关性，滞后响应）
    # SCFI在韩国出口先行指数变化后约2-4周会有响应
    scfi_base = np.array([3500 + 8 * korea_leading[i] + 20 * np.sin(i / 26 * 2 * np.pi) for i in range(num_weeks)])
    scfi_composite = scfi_base + np.random.randn(num_weeks) * 80

    # 生成SCFI各航线数据（与综合指数相关）
    scfi_data = {
        '日期': dates,
        '综合指数': scfi_composite,
        '欧洲航线': scfi_composite * 1.5 + np.random.randn(num_weeks) * 120,
        '地中海航线': scfi_composite * 1.4 + np.random.randn(num_weeks) * 110,
        '美西航线': scfi_composite * 1.6 + np.random.randn(num_weeks) * 130,
        '西非航线': scfi_composite * 1.3 + np.random.randn(num_weeks) * 100,
        '南美航线': scfi_composite * 1.9 + np.random.randn(num_weeks) * 150,
        '美东航线': scfi_composite * 2.2 + np.random.randn(num_weeks) * 170,
        '东南亚航线': scfi_composite * 0.28 + np.random.randn(num_weeks) * 25,
        '韩国航线': scfi_composite * 0.08 + np.random.randn(num_weeks) * 8,
    }
    scfi_df = pd.DataFrame(scfi_data)

    # 生成韩国出口指数数据（各指数间有相关性）
    korea_data = {
        '日期': dates,
        '韩国出口综合景气指数': korea_leading + np.random.randn(num_weeks) * 3,
        '对中国出口指数': korea_leading * 0.92 + np.random.randn(num_weeks) * 4,
        '对美国出口指数': korea_leading * 1.08 + np.random.randn(num_weeks) * 4,
        '对欧盟出口指数': korea_leading * 0.95 + np.random.randn(num_weeks) * 4,
        '对东南亚出口指数': korea_leading * 0.88 + np.random.randn(num_weeks) * 4,
        '韩国出口先行指数': korea_leading + np.random.randn(num_weeks) * 2,
    }
    korea_df = pd.DataFrame(korea_data)

    return scfi_df, korea_df

def save_mock_data(scfi_df, korea_df, output_dir='.'):
    """保存模拟数据到Excel文件（模拟真实数据结构，带标题行）

    Args:
        scfi_df: SCFI数据
        korea_df: 韩国出口指数数据
        output_dir: 输出目录
    """
    scfi_path = f'{output_dir}/上海出口集装箱运价指数.xlsx'
    korea_path = f'{output_dir}/韩国出口指数.xlsx'

    # SCFI数据列名
    scfi_columns = ['日期', '综合指数', '欧洲航线', '地中海航线', '美西航线', '西非航线', '南美航线', '美东航线', '东南亚航线', '韩国航线']

    # 韩国出口数据列名
    korea_columns = ['日期', '韩国出口综合景气指数', '对中国出口指数', '对美国出口指数', '对欧盟出口指数', '对东南亚出口指数', '韩国出口先行指数']

    # 设置SCFI DataFrame的列名
    scfi_df.columns = scfi_columns

    # 写入SCFI Excel
    # to_excel会用DataFrame的列名作为标题行，读取时skiprows=1跳过头部
    with pd.ExcelWriter(scfi_path, engine='openpyxl') as writer:
        scfi_df.to_excel(writer, index=False, header=True, startrow=0)
    print(f'SCFI数据已保存至: {scfi_path}')

    # 设置韩国出口 DataFrame的列名
    korea_df.columns = korea_columns

    # 写入韩国出口指数 Excel（带2行标题）
    with pd.ExcelWriter(korea_path, engine='openpyxl') as writer:
        # 第一行：第一组标题（空的第一列）
        pd.DataFrame([[''] + ['韩国出口综合景气指数']*6]).to_excel(writer, index=False, header=False, startrow=0)
        # 第二行：列名（作为第二级标题，与数据对应）
        pd.DataFrame([korea_columns]).to_excel(writer, index=False, header=False, startrow=1)
        # 数据从第3行开始
        korea_df.to_excel(writer, index=False, header=False, startrow=2)
    print(f'韩国出口指数已保存至: {korea_path}')

if __name__ == '__main__':
    # 生成模拟数据（500周，约10年历史数据）
    print('正在生成模拟数据...')
    scfi_df, korea_df = generate_mock_data(num_weeks=500)

    print(f'SCFI数据形状: {scfi_df.shape}')
    print(f'韩国出口指数数据形状: {korea_df.shape}')

    # 显示数据前几行
    print('\nSCFI数据预览:')
    print(scfi_df.head())

    print('\n韩国出口指数预览:')
    print(korea_df.head())

    # 保存到data_source目录
    import os
    data_source_dir = os.path.join(os.path.dirname(__file__), 'data_source')
    os.makedirs(data_source_dir, exist_ok=True)

    save_mock_data(scfi_df, korea_df, data_source_dir)
    print('\n模拟数据生成完成！')

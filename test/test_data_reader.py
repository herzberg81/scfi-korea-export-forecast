"""
DataReader类的单元测试
测试数据读取功能
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.DataReader import DataReader


class TestDataReader(unittest.TestCase):
    """DataReader类的测试用例"""

    def setUp(self):
        """测试前的准备工作
        创建临时测试数据文件
        """
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()

        # 创建测试用的SCFI数据
        scfi_data = {
            '日期': pd.date_range('2023-01-01', periods=10, freq='W'),
            '综合指数': [1000.0, 1010.0, 1020.0, 1030.0, 1040.0,
                           1050.0, 1060.0, 1070.0, 1080.0, 1090.0],
            '欧洲航线': [1500.0, 1510.0, 1520.0, 1530.0, 1540.0,
                      1550.0, 1560.0, 1570.0, 1580.0, 1590.0]
        }
        self.scfi_df = pd.DataFrame(scfi_data)

        # 创建测试用的韩国出口数据
        korea_data = {
            '日期': pd.date_range('2023-01-01', periods=10, freq='W'),
            '韩国出口综合景气指数': [90.5, 91.0, 92.0, 93.5, 94.0,
                           95.0, 96.5, 97.0, 98.0, 99.5]
        }
        self.korea_df = pd.DataFrame(korea_data)

        # 保存测试数据到临时文件
        self.scfi_file = Path(self.temp_dir) / 'test_scfi.xlsx'
        self.korea_file = Path(self.temp_dir) / 'test_korea.xlsx'

        # 写入Excel文件（直接写入，第一行就是列名，与实际数据格式一致）
        with pd.ExcelWriter(self.scfi_file, engine='openpyxl') as writer:
            self.scfi_df.to_excel(writer, index=False, header=True)

        with pd.ExcelWriter(self.korea_file, engine='openpyxl') as writer:
            # 韩国出口指数文件跳过第一行标题（模拟实际数据格式）
            pd.DataFrame([['韩国出口综合指数']]).to_excel(writer, index=False, header=False)
            self.korea_df.to_excel(writer, index=False, header=True, startrow=1)

        # 创建DataReader实例
        self.reader = DataReader(scfi_file=self.scfi_file, korea_export_file=self.korea_file)

    def tearDown(self):
        """测试后的清理工作
        删除临时文件和目录
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """测试初始化方法"""
        self.assertIsNotNone(self.reader)
        self.assertEqual(self.reader.scfi_file, self.scfi_file)
        self.assertEqual(self.reader.korea_export_file, self.korea_file)
        print("[OK] 测试DataReader初始化成功")

    def test_read_scfi_excel(self):
        """测试读取SCFI Excel文件"""
        # 读取数据
        df = self.reader.read_scfi_excel()

        # 验证数据不为空
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        print(f"[OK] 测试读取SCFI数据成功，共 {len(df)} 行")

    def test_read_korea_export_excel(self):
        """测试读取韩国出口指数Excel文件"""
        # 读取数据
        df = self.reader.read_korea_export_excel()

        # 验证数据不为空
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        print(f"[OK] 测试读取韩国出口数据成功，共 {len(df)} 行")

    def test_get_original_two_index(self):
        """测试获取两个原始指数数据"""
        # 获取两个指数数据
        scfi_df, korea_df = self.reader.get_original_two_index()

        # 验证返回的数据不为空
        self.assertIsNotNone(scfi_df)
        self.assertIsNotNone(korea_df)
        self.assertGreater(len(scfi_df), 0)
        self.assertGreater(len(korea_df), 0)
        print(f"[OK] 测试获取两个指数数据成功，SCFI {len(scfi_df)} 行，韩国出口 {len(korea_df)} 行")

    def test_file_not_found(self):
        """测试文件不存在时的异常处理"""
        # 创建一个不存在的文件路径
        fake_file = Path(self.temp_dir) / 'fake_file.xlsx'
        reader = DataReader(scfi_file=fake_file, korea_export_file=self.korea_file)

        # 测试文件不存在时抛出FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            reader.read_scfi_excel()
        print("[OK] 测试文件不存在异常处理成功")

    def test_display_data_info(self):
        """测试显示数据基本信息方法"""
        # 读取数据
        scfi_df, _ = self.reader.get_original_two_index()

        # 测试显示数据信息（不应该抛出异常）
        try:
            self.reader.display_data_info(scfi_df, "测试数据")
            print("[OK] 测试显示数据信息成功")
        except Exception as e:
            self.fail(f"显示数据信息失败: {e}")


class TestDataReaderEdgeCases(unittest.TestCase):
    """DataReader类的边界情况测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_dataframe(self):
        """测试空数据框的处理"""
        # 创建空数据框
        empty_df = pd.DataFrame()

        # 创建DataReader实例
        reader = DataReader()

        # 测试显示空数据信息，空数据框会抛出ValueError异常
        with self.assertRaises(ValueError):
            reader.display_data_info(empty_df, "空数据")
        print("[OK] 测试空数据框异常处理成功")

    def test_custom_parameters(self):
        """测试自定义参数"""
        # 创建测试数据
        scfi_file = Path(self.temp_dir) / 'custom_scfi.xlsx'
        korea_file = Path(self.temp_dir) / 'custom_korea.xlsx'

        # 创建简单的测试数据（使用中文列名，与实际数据格式一致）
        df = pd.DataFrame({
            '日期': pd.date_range('2023-01-01', periods=5, freq='W'),
            '综合指数': [100, 101, 102, 103, 104]
        })

        # 写入文件
        with pd.ExcelWriter(scfi_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        # 韩国出口数据模拟实际格式（带一行标题）
        with pd.ExcelWriter(korea_file, engine='openpyxl') as writer:
            pd.DataFrame([['韩国出口综合指数']]).to_excel(writer, index=False, header=False)
            df.to_excel(writer, index=False, startrow=1)

        # 创建DataReader实例
        reader = DataReader(scfi_file=scfi_file, korea_export_file=korea_file)

        # 读取数据
        scfi_data = reader.read_scfi_excel()
        korea_data = reader.read_korea_export_excel()

        # 验证
        self.assertEqual(len(scfi_data), 5)
        self.assertEqual(len(korea_data), 5)
        print("[OK] 测试自定义参数成功")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
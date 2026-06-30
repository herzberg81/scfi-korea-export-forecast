"""
DataFusionPreprocessor类的单元测试
测试数据融合和预处理功能
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

from src.data.DataFusionPreprocessor import DataFusionPreprocessor


class TestDataFusionPreprocessor(unittest.TestCase):
    """DataFusionPreprocessor类的测试用例"""

    def setUp(self):
        """测试前的准备工作
        创建测试数据
        """
        # 创建预处理器实例
        self.preprocessor = DataFusionPreprocessor(lag_weeks=4, train_ratio=0.8)

        # 创建测试用的SCFI数据
        self.scfi_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20, freq='W'),
            'scfi_current': np.random.uniform(1000, 1100, 20),
            'europe': np.random.uniform(1500, 1600, 20),
            'mediterranean': np.random.uniform(1300, 1400, 20)
        })

        # 创建测试用的韩国出口数据
        self.korea_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20, freq='W'),
            'korea_export': np.random.uniform(90, 100, 20)
        })

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """测试初始化方法"""
        self.assertIsNotNone(self.preprocessor)
        self.assertEqual(self.preprocessor.lag_weeks, 4)
        self.assertEqual(self.preprocessor.train_ratio, 0.8)
        print("[OK] 测试DataFusionPreprocessor初始化成功")

    def test_merge_two_index(self):
        """测试合并两个指数数据"""
        # 合并数据
        merged_df = self.preprocessor.merge_two_index(
            self.scfi_df,
            self.korea_df,
            date_col_scfi='date',
            date_col_korea='date'
        )

        # 验证合并后的数据
        self.assertIsNotNone(merged_df)
        self.assertGreater(len(merged_df), 0)
        self.assertIn('date', merged_df.columns)
        self.assertIn('scfi_current', merged_df.columns)
        self.assertIn('korea_export', merged_df.columns)
        print(f"[OK] 测试合并数据成功，合并后 {len(merged_df)} 行，{len(merged_df.columns)} 列")

    def test_clean_data_remove_duplicates(self):
        """测试删除重复行"""
        # 创建包含重复行的数据
        df_with_dup = pd.concat([self.scfi_df, self.scfi_df.iloc[:5]], ignore_index=True)

        # 清洗数据
        cleaned_df = self.preprocessor.clean_data(df_with_dup, remove_duplicates=True)

        # 验证重复行已删除
        self.assertEqual(len(cleaned_df), len(self.scfi_df))
        print(f"[OK] 测试删除重复行成功，删除了 {len(df_with_dup) - len(cleaned_df)} 行")

    def test_clean_data_fill_missing_ffill(self):
        """测试前向填充缺失值"""
        # 创建包含缺失值的数据
        df_with_nan = self.scfi_df.copy()
        df_with_nan.loc[5:7, 'scfi_current'] = np.nan

        # 使用前向填充
        cleaned_df = self.preprocessor.clean_data(
            df_with_nan,
            remove_duplicates=False,
            fill_method='ffill'
        )

        # 验证没有缺失值
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
        print("[OK] 测试前向填充缺失值成功")

    def test_clean_data_fill_missing_interpolate(self):
        """测试线性插值填充缺失值"""
        # 创建包含缺失值的数据
        df_with_nan = self.scfi_df.copy()
        df_with_nan.loc[5:7, 'scfi_current'] = np.nan

        # 使用线性插值
        cleaned_df = self.preprocessor.clean_data(
            df_with_nan,
            remove_duplicates=False,
            fill_method='interpolate'
        )

        # 验证没有缺失值
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
        print("[OK] 测试线性插值填充缺失值成功")

    def test_clean_data_drop_missing(self):
        """测试删除缺失值"""
        # 创建包含缺失值的数据
        df_with_nan = self.scfi_df.copy()
        df_with_nan.loc[5:7, 'scfi_current'] = np.nan

        # 使用删除方式
        cleaned_df = self.preprocessor.clean_data(
            df_with_nan,
            remove_duplicates=False,
            fill_method='drop'
        )

        # 验证缺失值已删除
        self.assertLess(len(cleaned_df), len(df_with_nan))
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
        print(f"[OK] 测试删除缺失值成功，删除了 {len(df_with_nan) - len(cleaned_df)} 行")

    def test_build_lag_feature(self):
        """测试构建滞后特征"""
        # 先合并数据
        merged_df = self.preprocessor.merge_two_index(
            self.scfi_df,
            self.korea_df,
            date_col_scfi='date',
            date_col_korea='date'
        )

        # 清洗数据
        cleaned_df = self.preprocessor.clean_data(merged_df)

        # 构建滞后特征
        featured_df = self.preprocessor.build_lag_feature(
            cleaned_df,
            target_col='scfi_current'
        )

        # 验证滞后特征已创建
        self.assertIn('scfi_next', featured_df.columns)
        # 检查是否存在滞后特征列
        lag_cols = [col for col in featured_df.columns if 'lag' in col.lower()]
        self.assertGreater(len(lag_cols), 0)
        print(f"[OK] 测试构建滞后特征成功，创建了 {len(lag_cols)} 个滞后特征列")

    def test_split_dataset(self):
        """测试划分训练集和验证集"""
        # 先合并和处理数据
        merged_df = self.preprocessor.merge_two_index(
            self.scfi_df,
            self.korea_df,
            date_col_scfi='date',
            date_col_korea='date'
        )
        cleaned_df = self.preprocessor.clean_data(merged_df)
        featured_df = self.preprocessor.build_lag_feature(cleaned_df)

        # 划分数据集
        train_df, val_df = self.preprocessor.split_dataset(featured_df, train_ratio=0.8)

        # 验证划分比例
        total_len = len(featured_df)
        self.assertEqual(len(train_df) + len(val_df), total_len)
        self.assertAlmostEqual(len(train_df) / total_len, 0.8, places=1)
        print(f"[OK] 测试划分数据集成功，训练集 {len(train_df)} 行，验证集 {len(val_df)} 行")

    def test_split_dataset_custom_ratio(self):
        """测试自定义比例划分数据集"""
        # 先合并和处理数据
        merged_df = self.preprocessor.merge_two_index(
            self.scfi_df,
            self.korea_df,
            date_col_scfi='date',
            date_col_korea='date'
        )
        cleaned_df = self.preprocessor.clean_data(merged_df)
        featured_df = self.preprocessor.build_lag_feature(cleaned_df)

        # 使用自定义比例划分
        train_df, val_df = self.preprocessor.split_dataset(featured_df, train_ratio=0.7)

        # 验证划分比例
        total_len = len(featured_df)
        self.assertAlmostEqual(len(train_df) / total_len, 0.7, places=1)
        print(f"[OK] 测试自定义比例划分成功，比例 0.7")

    def test_detect_date_column(self):
        """测试检测日期列"""
        # 测试自动检测日期列
        date_col = self.preprocessor._detect_date_column(self.scfi_df, "测试")

        # 验证检测到的日期列
        self.assertEqual(date_col, 'date')
        print(f"[OK] 测试检测日期列成功，检测到: {date_col}")

    def test_convert_date_column(self):
        """测试转换日期列"""
        # 创建包含字符串日期的数据框
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-08', '2023-01-15'],
            'value': [100, 101, 102]
        })

        # 转换日期列
        converted_df = self.preprocessor._convert_date_column(df, 'date')

        # 验证日期列已转换为datetime类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted_df['date']))
        print("[OK] 测试转换日期列成功")

    def test_save_processed_data(self):
        """测试保存处理后的数据"""
        # 准备测试数据
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='W'),
            'scfi_current': np.random.uniform(1000, 1100, 10)
        })
        val_df = pd.DataFrame({
            'date': pd.date_range('2023-03-12', periods=5, freq='W'),
            'scfi_current': np.random.uniform(1000, 1100, 5)
        })

        # 保存数据
        train_path, val_path = self.preprocessor.save_processed_data(
            train_df,
            val_df,
            output_dir=Path(self.temp_dir)
        )

        # 验证文件已创建
        self.assertTrue(train_path.exists())
        self.assertTrue(val_path.exists())

        # 验证文件内容
        loaded_train = pd.read_csv(train_path)
        loaded_val = pd.read_csv(val_path)

        self.assertEqual(len(loaded_train), len(train_df))
        self.assertEqual(len(loaded_val), len(val_df))
        print("[OK] 测试保存处理后的数据成功")


class TestDataFusionPreprocessorEdgeCases(unittest.TestCase):
    """DataFusionPreprocessor类的边界情况测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.preprocessor = DataFusionPreprocessor(lag_weeks=3, train_ratio=0.75)

    def test_merge_with_different_date_ranges(self):
        """测试合并不同日期范围的数据"""
        # 创建日期范围不完全重叠的数据
        scfi_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=15, freq='W'),
            'scfi_current': np.random.uniform(1000, 1100, 15)
        })

        korea_df = pd.DataFrame({
            'date': pd.date_range('2023-01-15', periods=15, freq='W'),
            'korea_export': np.random.uniform(90, 100, 15)
        })

        # 合并数据
        merged_df = self.preprocessor.merge_two_index(
            scfi_df,
            korea_df,
            date_col_scfi='date',
            date_col_korea='date'
        )

        # 验证合并后的数据行数正确（内连接应该取交集）
        self.assertGreater(len(merged_df), 0)
        print(f"[OK] 测试合并不同日期范围数据成功，合并后 {len(merged_df)} 行")

    def test_clean_data_remove_outliers(self):
        """测试删除异常值"""
        # 创建包含异常值的数据
        df_with_outliers = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20, freq='W'),
            'scfi_current': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                           110, 111, 112, 113, 114, 115, 116, 117, 118, 5000]  # 最后一个是异常值
        })

        # 删除异常值
        cleaned_df = self.preprocessor.clean_data(
            df_with_outliers,
            remove_duplicates=False,
            fill_method='drop',
            remove_outliers=True,
            outlier_threshold=3.0
        )

        # 验证异常值已删除（数据行数应该减少）
        self.assertLess(len(cleaned_df), len(df_with_outliers))
        print(f"[OK] 测试删除异常值成功，删除了 {len(df_with_outliers) - len(cleaned_df)} 行")

    def test_build_lag_feature_with_custom_cols(self):
        """测试使用自定义列构建滞后特征"""
        # 创建测试数据
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='W'),
            'scfi_current': np.random.uniform(1000, 1100, 10),
            'korea_export': np.random.uniform(90, 100, 10)
        })

        # 只为指定列构建滞后特征
        featured_df = self.preprocessor.build_lag_feature(
            df,
            target_col='scfi_current',
            lag_cols=['scfi_current']  # 只为scfi_current构建滞后特征
        )

        # 验证只有scfi_current的滞后特征
        scfi_lag_cols = [col for col in featured_df.columns if 'scfi_current' in col and 'lag' in col]
        korea_lag_cols = [col for col in featured_df.columns if 'korea_export' in col and 'lag' in col]

        self.assertGreater(len(scfi_lag_cols), 0)
        self.assertEqual(len(korea_lag_cols), 0)  # korea_export不应该有滞后特征
        print(f"[OK] 测试自定义列构建滞后特征成功")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
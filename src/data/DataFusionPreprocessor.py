"""
数据融合预处理类
用于合并、清洗数据并构建滞后特征
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from datetime import datetime

# 获取模块logger（不在此处配置，由main.py统一配置）
logger = logging.getLogger(__name__)


class DataFusionPreprocessor:
    """数据融合预处理类

    用于合并SCFI和韩国出口指数数据，清洗数据，构建滞后特征，划分训练验证集
    """

    def __init__(self, lag_weeks: int = 4, train_ratio: float = 0.8):
        """初始化DataFusionPreprocessor

        Args:
            lag_weeks: 滞后周数，默认为4周
            train_ratio: 训练集比例，默认为0.8
        """
        from src.config.train_config import TrainConfig

        self.lag_weeks = lag_weeks if lag_weeks else TrainConfig.LAG_WEEKS
        self.train_ratio = train_ratio if train_ratio else TrainConfig.TRAIN_RATIO

        # 日期列名可能的候选
        self.date_columns = ['date', 'Date', '日期', '时间', 'week', 'Week']

        logger.info(f"DataFusionPreprocessor初始化完成")
        logger.info(f"滞后周数: {self.lag_weeks}")
        logger.info(f"训练集比例: {self.train_ratio}")

    def merge_two_index(
        self,
        scfi_df: pd.DataFrame,
        korea_export_df: pd.DataFrame,
        date_col_scfi: Optional[str] = None,
        date_col_korea: Optional[str] = None,
        how: str = 'inner'
    ) -> pd.DataFrame:
        """按日期合并两个指数数据

        Args:
            scfi_df: SCFI数据框
            korea_export_df: 韩国出口数据框
            date_col_scfi: SCFI数据的日期列名，如未指定则自动检测
            date_col_korea: 韩国出口数据的日期列名，如未指定则自动检测
            how: 合并方式，默认为'inner'（内连接）

        Returns:
            pd.DataFrame: 合并后的数据框

        Raises:
            ValueError: 找不到日期列时抛出异常
        """
        logger.info("=" * 50)
        logger.info("开始合并两个指数数据")
        logger.info("=" * 50)

        # 自动检测日期列
        if date_col_scfi is None:
            date_col_scfi = self._detect_date_column(scfi_df, "SCFI")
        if date_col_korea is None:
            date_col_korea = self._detect_date_column(korea_export_df, "韩国出口")

        logger.info(f"SCFI日期列: {date_col_scfi}")
        logger.info(f"韩国出口日期列: {date_col_korea}")

        # 转换日期列为datetime类型
        scfi_df = self._convert_date_column(scfi_df, date_col_scfi)
        korea_export_df = self._convert_date_column(korea_export_df, date_col_korea)

        # 重命名日期列以便合并
        scfi_df_temp = scfi_df.copy()
        korea_export_df_temp = korea_export_df.copy()

        # 为列添加前缀以区分来源
        # SCFI列保留原样，韩国出口列添加前缀
        korea_export_cols = [col for col in korea_export_df_temp.columns if col != date_col_korea]

        # 合并数据
        merged_df = pd.merge(
            scfi_df_temp,
            korea_export_df_temp,
            left_on=date_col_scfi,
            right_on=date_col_korea,
            how=how
        )

        # 统一日期列名
        if date_col_scfi in merged_df.columns:
            merged_df.rename(columns={date_col_scfi: 'date'}, inplace=True)
        elif date_col_korea in merged_df.columns:
            merged_df.rename(columns={date_col_korea: 'date'}, inplace=True)

        # 如果存在两个日期列，删除其中一个
        if date_col_korea in merged_df.columns and date_col_korea != 'date':
            merged_df.drop(columns=[date_col_korea], inplace=True)

        # 按日期排序
        merged_df.sort_values(by='date', inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        logger.info(f"合并完成，共 {len(merged_df)} 行数据")
        logger.info(f"列名: {merged_df.columns.tolist()}")

        return merged_df

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        fill_method: str = 'ffill',
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0
    ) -> pd.DataFrame:
        """清洗数据

        Args:
            df: 待清洗的数据框
            remove_duplicates: 是否删除重复行，默认为True
            fill_method: 缺失值填充方法，可选 'ffill'(前向填充), 'bfill'(后向填充), 'interpolate'(插值), 'drop'(删除)
            remove_outliers: 是否删除异常值，默认为False
            outlier_threshold: 异常值阈值（标准差倍数），默认为3.0

        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        logger.info("=" * 50)
        logger.info("开始数据清洗")
        logger.info("=" * 50)

        cleaned_df = df.copy()

        logger.info(f"原始数据: {len(cleaned_df)} 行")

        # 1. 删除重复行
        if remove_duplicates:
            before_dup = len(cleaned_df)
            cleaned_df.drop_duplicates(inplace=True)
            after_dup = len(cleaned_df)
            if before_dup != after_dup:
                logger.info(f"删除重复行: {before_dup - after_dup} 行")

        # 2. 处理缺失值
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"发现缺失值: {missing_count} 个")

            if fill_method == 'drop':
                before_drop = len(cleaned_df)
                cleaned_df.dropna(inplace=True)
                after_drop = len(cleaned_df)
                logger.info(f"删除缺失值行: {before_drop - after_drop} 行")
            elif fill_method == 'ffill':
                cleaned_df = cleaned_df.ffill()
                cleaned_df = cleaned_df.bfill()  # 如果开头有缺失值，用后向填充
                logger.info("使用前向填充处理缺失值")
            elif fill_method == 'bfill':
                cleaned_df = cleaned_df.bfill()
                cleaned_df = cleaned_df.ffill()  # 如果结尾有缺失值，用前向填充
                logger.info("使用后向填充处理缺失值")
            elif fill_method == 'interpolate':
                # 对数值列进行线性插值
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].interpolate(method='linear')
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].ffill()
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].bfill()
                logger.info("使用线性插值处理缺失值")
            else:
                logger.warning(f"未知的填充方法: {fill_method}，跳过缺失值处理")

        # 3. 删除异常值
        if remove_outliers:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            before_outlier = len(cleaned_df)

            for col in numeric_cols:
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                if std > 0:  # 避免除以0
                    # 使用Z-score方法检测异常值
                    z_scores = np.abs((cleaned_df[col] - mean) / std)
                    cleaned_df = cleaned_df[z_scores <= outlier_threshold]

            after_outlier = len(cleaned_df)
            if before_outlier != after_outlier:
                logger.info(f"删除异常值行: {before_outlier - after_outlier} 行")

        logger.info(f"清洗后数据: {len(cleaned_df)} 行")
        logger.info("=" * 50)

        return cleaned_df

    def build_lag_feature(
        self,
        df: pd.DataFrame,
        target_col: str = 'scfi_current',
        lag_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """构造滞后特征

        Args:
            df: 数据框
            target_col: 目标列名（需要预测的SCFI下一周指数）
            lag_cols: 需要构建滞后特征的列名列表，如未指定则自动选择数值列

        Returns:
            pd.DataFrame: 包含滞后特征的数据框
        """
        logger.info("=" * 50)
        logger.info("开始构建滞后特征")
        logger.info("=" * 50)

        result_df = df.copy()

        # 如果未指定滞后列，则选择所有数值列（排除日期列）
        if lag_cols is None:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            lag_cols = [col for col in numeric_cols if col != 'date']
            logger.info(f"自动选择滞后特征列: {lag_cols}")

        # 确保数据按日期排序
        if 'date' in result_df.columns:
            result_df.sort_values('date', inplace=True)
            result_df.reset_index(drop=True, inplace=True)

        # 为每个指定列构建滞后特征
        for col in lag_cols:
            if col not in result_df.columns:
                logger.warning(f"列 '{col}' 不存在，跳过")
                continue

            for lag in range(1, self.lag_weeks + 1):
                lag_col_name = f"{col}_lag{lag}"
                result_df[lag_col_name] = result_df[col].shift(lag)
                logger.info(f"创建滞后特征: {lag_col_name}")

        # 构建目标列（下一周的SCFI指数）
        if target_col in result_df.columns:
            result_df['scfi_next'] = result_df[target_col].shift(-1)
            logger.info(f"创建目标列: scfi_next (下一周的SCFI指数)")

        # 删除因构建滞后特征而产生的缺失值行
        before_drop = len(result_df)
        result_df.dropna(inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        after_drop = len(result_df)

        if before_drop != after_drop:
            logger.info(f"删除因滞后特征产生的缺失值行: {before_drop - after_drop} 行")

        logger.info(f"滞后特征构建完成，最终数据: {len(result_df)} 行，{len(result_df.columns)} 列")
        logger.info("=" * 50)

        return result_df

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: Optional[float] = None,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和验证集

        Args:
            df: 数据框
            train_ratio: 训练集比例，如未指定则使用初始化时的值
            shuffle: 是否打乱数据，默认为False（保持时间序列顺序）

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 验证集)
        """
        logger.info("=" * 50)
        logger.info("开始划分数据集")
        logger.info("=" * 50)

        if train_ratio is None:
            train_ratio = self.train_ratio

        logger.info(f"训练集比例: {train_ratio}")

        # 对于时间序列数据，通常不打乱顺序
        if shuffle:
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df_shuffled = df.copy()

        # 计算分割点
        split_index = int(len(df_shuffled) * train_ratio)

        # 划分数据集
        train_df = df_shuffled.iloc[:split_index].copy()
        val_df = df_shuffled.iloc[split_index:].copy()

        logger.info(f"训练集大小: {len(train_df)} 行")
        logger.info(f"验证集大小: {len(val_df)} 行")
        logger.info("=" * 50)

        return train_df, val_df

    def _detect_date_column(self, df: pd.DataFrame, dataset_name: str) -> str:
        """检测日期列

        Args:
            df: 数据框
            dataset_name: 数据集名称（用于日志）

        Returns:
            str: 日期列名

        Raises:
            ValueError: 找不到日期列时抛出异常
        """
        for col in self.date_columns:
            if col in df.columns:
                logger.info(f"{dataset_name}数据检测到日期列: {col}")
                return col

        # 如果没找到，尝试查找包含日期信息的列
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试解析日期
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    logger.info(f"{dataset_name}数据检测到日期列（通过解析）: {col}")
                    return col
                except:
                    pass

        raise ValueError(f"在{dataset_name}数据中找不到日期列，请明确指定日期列名")

    def _convert_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """转换日期列为datetime类型

        Args:
            df: 数据框
            date_col: 日期列名

        Returns:
            pd.DataFrame: 转换后的数据框
        """
        df = df.copy()

        # 如果日期列不是datetime类型，则转换
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # 删除无法解析的日期行
        before_drop = len(df)
        df.dropna(subset=[date_col], inplace=True)
        after_drop = len(df)

        if before_drop != after_drop:
            logger.info(f"删除无效日期行: {before_drop - after_drop} 行")

        return df

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """保存处理后的训练和验证数据

        Args:
            train_df: 训练集数据框
            val_df: 验证集数据框
            output_dir: 输出目录，如未指定则使用配置文件中的默认目录

        Returns:
            Tuple[Path, Path]: (训练数据文件路径, 验证数据文件路径)
        """
        from src.config.train_config import TrainConfig

        if output_dir is None:
            output_dir = TrainConfig.PROCESSED_DATA_DIR

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        train_path = output_dir / 'train_data.csv'
        val_path = output_dir / 'val_data.csv'

        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_df.to_csv(val_path, index=False, encoding='utf-8-sig')

        logger.info(f"训练数据已保存至: {train_path}")
        logger.info(f"验证数据已保存至: {val_path}")

        return train_path, val_path


# 示例使用
if __name__ == '__main__':
    from src.data.DataReader import DataReader

    try:
        # 读取数据
        reader = DataReader()
        scfi_df, korea_df = reader.get_original_two_index()

        # 创建预处理器
        preprocessor = DataFusionPreprocessor(lag_weeks=4, train_ratio=0.8)

        # 合并数据
        merged_df = preprocessor.merge_two_index(scfi_df, korea_df)
        print("\n合并后数据:")
        print(merged_df.head())
        print(f"形状: {merged_df.shape}")

        # 清洗数据
        cleaned_df = preprocessor.clean_data(merged_df)
        print("\n清洗后数据:")
        print(cleaned_df.head())
        print(f"形状: {cleaned_df.shape}")

        # 构建滞后特征
        featured_df = preprocessor.build_lag_feature(cleaned_df, target_col='scfi_current')
        print("\n构建滞后特征后数据:")
        print(featured_df.head())
        print(f"形状: {featured_df.shape}")
        print(f"列名: {featured_df.columns.tolist()}")

        # 划分数据集
        train_df, val_df = preprocessor.split_dataset(featured_df)
        print("\n训练集:")
        print(train_df.head())
        print(f"形状: {train_df.shape}")
        print("\n验证集:")
        print(val_df.head())
        print(f"形状: {val_df.shape}")

        # 保存数据
        train_path, val_path = preprocessor.save_processed_data(train_df, val_df)

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保数据文件存在")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
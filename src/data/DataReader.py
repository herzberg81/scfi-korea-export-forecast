"""
数据读取类
用于读取SCFI指数和韩国出口指数的Excel数据
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

# 获取模块logger（不在此处配置，由main.py统一配置）
logger = logging.getLogger(__name__)


class DataReader:
    """数据读取类

    用于读取SCFI指数和韩国出口指数的Excel数据文件
    """

    def __init__(self, scfi_file: Optional[Path] = None, korea_export_file: Optional[Path] = None):
        """初始化DataReader

        Args:
            scfi_file: SCFI数据文件路径，如未指定则使用配置文件中的默认路径
            korea_export_file: 韩国出口数据文件路径，如未指定则使用配置文件中的默认路径
        """
        from src.config.train_config import TrainConfig

        self.scfi_file = scfi_file if scfi_file else TrainConfig.SCFI_DATA_FILE
        self.korea_export_file = korea_export_file if korea_export_file else TrainConfig.KOREA_EXPORT_DATA_FILE

        logger.info(f"DataReader初始化完成")
        logger.info(f"SCFI数据文件: {self.scfi_file}")
        logger.info(f"韩国出口数据文件: {self.korea_export_file}")

    def read_scfi_excel(self, sheet_name: int = 0) -> pd.DataFrame:
        """读取SCFI指数Excel数据

        Args:
            sheet_name: 工作表名称或索引，默认为第一个工作表

        Returns:
            pd.DataFrame: SCFI指数数据

        Raises:
            FileNotFoundError: 文件不存在时抛出异常
            Exception: 读取文件失败时抛出异常
        """
        try:
            # 检查文件是否存在
            if not self.scfi_file.exists():
                raise FileNotFoundError(f"SCFI数据文件不存在: {self.scfi_file}")

            logger.info(f"开始读取SCFI数据: {self.scfi_file}")

            # 读取Excel文件，header=0表示第一行是列名
            df = pd.read_excel(
                self.scfi_file,
                sheet_name=sheet_name,
                engine='openpyxl'
            )

            logger.info(f"成功读取SCFI数据，共 {len(df)} 行，{len(df.columns)} 列")
            logger.info(f"列名: {df.columns.tolist()}")

            # 检查是否有数据
            if len(df) == 0:
                logger.warning("警告: SCFI数据为空!")

            return df

        except FileNotFoundError as e:
            logger.error(f"文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"读取SCFI数据失败: {e}")
            raise

    def read_korea_export_excel(self, sheet_name: int = 0) -> pd.DataFrame:
        """读取韩国出口指数Excel数据

        Args:
            sheet_name: 工作表名称或索引，默认为第一个工作表

        Returns:
            pd.DataFrame: 韩国出口指数数据

        Raises:
            FileNotFoundError: 文件不存在时抛出异常
            Exception: 读取文件失败时抛出异常
        """
        try:
            # 检查文件是否存在
            if not self.korea_export_file.exists():
                raise FileNotFoundError(f"韩国出口数据文件不存在: {self.korea_export_file}")

            logger.info(f"开始读取韩国出口数据: {self.korea_export_file}")

            # 读取Excel文件，跳过第一行分组标题，用第二行作为列名
            df = pd.read_excel(
                self.korea_export_file,
                sheet_name=sheet_name,
                skiprows=1,  # 跳过第一行分组标题，header默认是0
                engine='openpyxl'
            )

            logger.info(f"成功读取韩国出口数据，共 {len(df)} 行，{len(df.columns)} 列")
            logger.info(f"列名: {df.columns.tolist()}")

            # 检查是否有数据
            if len(df) == 0:
                logger.warning("警告: 韩国出口数据为空!")

            return df

        except FileNotFoundError as e:
            logger.error(f"文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"读取韩国出口数据失败: {e}")
            raise

    def get_original_two_index(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取两个原始指数数据

        读取SCFI指数和韩国出口指数数据

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (SCFI数据, 韩国出口数据)

        Raises:
            Exception: 读取数据失败时抛出异常
        """
        logger.info("=" * 50)
        logger.info("开始读取原始指数数据")
        logger.info("=" * 50)

        try:
            # 读取SCFI数据
            scfi_df = self.read_scfi_excel()

            # 读取韩国出口数据
            korea_export_df = self.read_korea_export_excel()

            logger.info("=" * 50)
            logger.info("原始指数数据读取完成")
            logger.info(f"SCFI数据: {len(scfi_df)} 行")
            logger.info(f"韩国出口数据: {len(korea_export_df)} 行")
            logger.info("=" * 50)

            return scfi_df, korea_export_df

        except Exception as e:
            logger.error(f"读取原始指数数据失败: {e}")
            raise

    def display_data_info(self, df: pd.DataFrame, name: str = "数据"):
        """显示数据基本信息

        Args:
            df: 数据框
            name: 数据名称
        """
        print(f"\n{name}信息:")
        print("=" * 50)
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"数据类型:\n{df.dtypes}")
        print(f"\n前5行数据:")
        print(df.head())
        print(f"\n数据统计描述:")
        print(df.describe())
        print("=" * 50)


# 示例使用
if __name__ == '__main__':
    try:
        # 创建DataReader实例
        reader = DataReader()

        # 读取数据（如果文件存在）
        scfi_df, korea_df = reader.get_original_two_index()

        # 显示数据信息
        reader.display_data_info(scfi_df, "SCFI数据")
        reader.display_data_info(korea_df, "韩国出口数据")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保数据文件存在于正确的位置:")
        print(f"  - SCFI文件: {reader.scfi_file}")
        print(f"  - 韩国出口文件: {reader.korea_export_file}")
    except Exception as e:
        print(f"发生错误: {e}")
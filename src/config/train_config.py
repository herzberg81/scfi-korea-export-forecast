"""
训练配置类
包含模型训练所需的各种配置参数
"""
import os
from pathlib import Path


class TrainConfig:
    """训练配置类，包含滞后周数、训练集比例、模型类型列表、文件路径配置"""

    # ==================== 数据处理参数 ====================
    # 滞后周数（使用前几周的数据预测下一周）
    LAG_WEEKS = 4

    # 训练集比例（80%训练，20%验证）
    TRAIN_RATIO = 0.8

    # ==================== 模型配置 ====================
    # 支持的模型类型列表
    MODEL_TYPES = [
        'linear',      # 线性回归
        'ridge',       # 岭回归
        'lasso',       # Lasso回归
        'elasticnet',  # 弹性网络
        'random_forest',  # 随机森林
        'gradient_boosting',  # 梯度提升
        'xgboost',     # XGBoost
        'lightgbm',    # LightGBM
        'svr',         # 支持向量回归
    ]

    # 默认模型参数
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        },
    }

    # ==================== 文件路径配置 ====================
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # 数据目录
    DATA_DIR = PROJECT_ROOT / 'data'

    # 原始数据目录（使用data_source而非data/raw）
    RAW_DATA_DIR = PROJECT_ROOT / 'data_source'

    # 处理后数据目录
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'

    # 模型保存目录
    MODEL_DIR = PROJECT_ROOT / 'models_save'

    # 结果输出目录
    OUTPUT_DIR = PROJECT_ROOT / 'output'

    # 日志目录
    LOG_DIR = PROJECT_ROOT / 'logs'

    # SCFI数据文件路径
    SCFI_DATA_FILE = RAW_DATA_DIR / '上海出口集装箱运价指数.xlsx'

    # 韩国出口指数数据文件路径
    KOREA_EXPORT_DATA_FILE = RAW_DATA_DIR / '韩国出口指数.xlsx'

    # 合并后的数据文件路径
    MERGED_DATA_FILE = PROCESSED_DATA_DIR / 'merged_data.csv'

    # 训练数据文件路径
    TRAIN_DATA_FILE = PROCESSED_DATA_DIR / 'train_data.csv'

    # 验证数据文件路径
    VAL_DATA_FILE = PROCESSED_DATA_DIR / 'val_data.csv'

    # ==================== 其他配置 ====================
    # 随机种子
    RANDOM_SEED = 42

    # 日志级别
    LOG_LEVEL = 'INFO'

    # 日期格式
    DATE_FORMAT = '%Y-%m-%d'

    # ==================== 兼兼容别名（供main.py使用） ====================
    # SCFI数据文件路径（兼容别名）
    SCFI_FILE = SCFI_DATA_FILE

    # 韩国出口指数数据文件路径（兼容别名）
    KOREA_FILE = KOREA_EXPORT_DATA_FILE

    # 模型保存目录（兼容别名）
    MODEL_SAVE_DIR = MODEL_DIR

    # 可预测航线列表
    ROUTES = ['综合指数', '欧洲航线', '地中海航线', '美西航线', '西非航线',
              '南美航线', '美东航线', '东南亚航线', '韩国航线']

    @classmethod
    def create_directories(cls):
        """创建必要的目录结构"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.OUTPUT_DIR,
            cls.LOG_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls, model_name):
        """获取模型保存路径

        Args:
            model_name: 模型名称

        Returns:
            Path: 模型文件路径
        """
        return cls.MODEL_DIR / f'{model_name}.pkl'

    @classmethod
    def get_result_path(cls, filename):
        """获取结果文件保存路径

        Args:
            filename: 文件名

        Returns:
            Path: 结果文件路径
        """
        return cls.OUTPUT_DIR / filename

    @classmethod
    def display_config(cls):
        """显示当前配置信息"""
        print("=" * 50)
        print("训练配置信息:")
        print("=" * 50)
        print(f"滞后周数: {cls.LAG_WEEKS}")
        print(f"训练集比例: {cls.TRAIN_RATIO}")
        print(f"支持的模型类型: {', '.join(cls.MODEL_TYPES)}")
        print(f"随机种子: {cls.RANDOM_SEED}")
        print(f"项目根目录: {cls.PROJECT_ROOT}")
        print(f"数据目录: {cls.DATA_DIR}")
        print(f"模型保存目录: {cls.MODEL_DIR}")
        print(f"结果输出目录: {cls.OUTPUT_DIR}")
        print("=" * 50)


# 如果直接运行此文件，显示配置信息
if __name__ == '__main__':
    TrainConfig.display_config()


# 提供 Config 别名以兼容旧代码
Config = TrainConfig
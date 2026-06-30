# -*- coding: utf-8 -*-
"""
SCFI 海运运价预测系统 - 主程序入口
基于韩国出口先行指数和SCFI上海集装箱运价指数的周度运价预测

功能流程：
    读取数据 → 预处理 → 训练模型 → 预测 → 分析 → 导出结果

使用方法：
    python main.py --route 综合指数
    python main.py --route 欧洲航线
    python main.py --help 查看所有航线选项
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 统一配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[logging.StreamHandler()]
)

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入项目模块
try:
    from src.data.DataReader import DataReader
    from src.data.DataFusionPreprocessor import DataFusionPreprocessor
    from src.model.SCFIForecastModel import SCFIForecastModel
    from src.analysis.ResultAnalyzer import ResultAnalyzer
    from src.export.ResultExporter import ResultExporter
    from src.config.train_config import Config
except ImportError as e:
    print(f"警告: 模块导入失败 - {e}")
    print("请确保所有依赖模块已正确创建")
    # 为了避免程序直接退出，创建占位符类
    print("使用占位符模式继续运行...")

    class DataReader:
        def __init__(self, *args, **kwargs): pass
        def read_scfi_excel(self): pass
        def read_korea_export_excel(self): pass
        def get_original_two_index(self): return None, None

    class DataFusionPreprocessor:
        def __init__(self, *args, **kwargs): pass
        def merge_two_index(self, *args): pass
        def clean_data(self): pass
        def build_lag_feature(self, *args): pass
        def split_dataset(self, *args): return None, None, None, None

    class SCFIForecastModel:
        def __init__(self, *args, **kwargs): pass
        def train(self, *args): pass
        def predict(self, *args): return None
        def save_model(self, *args): pass
        def get_feature_importance(self): return {}

    class ResultAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def calc_evaluation_metrics(self, *args): return {}
        def get_feature_importance(self, *args): return {}

    class ResultExporter:
        def __init__(self, *args, **kwargs): pass
        def draw_fit_plot(self, *args, **kwargs): pass
        def draw_feature_importance(self, *args, **kwargs): pass
        def export_pred_excel(self, *args, **kwargs): pass
        def export_log(self, *args, **kwargs): pass

    class Config:
        SCFI_FILE = "data_source/上海出口集装箱运价指数.xlsx"
        KOREA_FILE = "data_source/韩国出口指数.xlsx"
        LAG_WEEKS = 4
        TRAIN_RATIO = 0.8
        MODEL_SAVE_DIR = "models_save"
        OUTPUT_DIR = "output"
        ROUTES = ['综合指数', '欧洲航线', '地中海航线', '美西航线', '西非航线',
                  '南美航线', '美东航线', '东南亚航线', '韩国航线']


# 定义可预测的航线列表
AVAILABLE_ROUTES = [
    '综合指数',
    '欧洲航线',
    '地中海航线',
    '美西航线',
    '西非航线',
    '南美航线',
    '美东航线',
    '东南亚航线',
    '韩国航线'
]


def print_banner():
    """打印程序横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          SCFI 海运运价预测系统 v1.0                         ║
║   基于韩国出口先行指数 + SCFI 上海集装箱运价指数            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")


def safe_print(msg):
    """安全打印，避免编码问题"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('utf-8', errors='replace').decode('utf-8'))


def main(route_name='综合指数', model_type='random_forest'):
    """
    主函数：执行完整的预测流程

    Args:
        route_name: 要预测的航线名称，默认为'综合指数'
        model_type: 模型类型，支持 'random_forest', 'linear', 'xgboost'

    Returns:
        dict: 包含预测结果和评估指标的字典
    """
    # 打印程序横幅
    print_banner()
    print(f"预测航线: {route_name}")
    print(f"模型类型: {model_type}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # ========================
        # 第一步：读取数据
        # ========================
        print_separator("第一步：读取数据")

        # 初始化数据读取器
        config = Config()
        data_reader = DataReader(
            scfi_file=config.SCFI_FILE,
            korea_export_file=config.KOREA_FILE
        )

        print(f"正在读取 SCFI 数据: {config.SCFI_FILE}")
        data_reader.read_scfi_excel()

        print(f"正在读取韩国出口指数数据: {config.KOREA_FILE}")
        data_reader.read_korea_export_excel()

        # 获取原始数据
        scfi_df, korea_df = data_reader.get_original_two_index()
        print(f"[OK] SCFI 数据读取成功，共 {len(scfi_df)} 条记录")
        print(f"[OK] 韩国出口指数读取成功，共 {len(korea_df)} 条记录")

        # ========================
        # 第二步：数据预处理与融合
        # ========================
        print_separator("第二步：数据预处理与融合")

        # 初始化数据预处理器
        preprocessor = DataFusionPreprocessor(
            lag_weeks=config.LAG_WEEKS
        )

        # 合并两份指数数据
        print("正在合并 SCFI 与韩国出口指数数据...")
        merged_df = preprocessor.merge_two_index(
            scfi_df,
            korea_df,
            date_col_scfi='日期',
            date_col_korea='日期'
        )

        # 清洗数据
        print("正在进行数据清洗...")
        cleaned_df = preprocessor.clean_data(merged_df)

        # 构建滞后特征
        print(f"正在构建滞后特征（滞后{config.LAG_WEEKS}周）...")
        featured_df = preprocessor.build_lag_feature(cleaned_df, target_col='综合指数')

        # 划分训练集和测试集
        print(f"正在划分数据集（训练集比例: {config.TRAIN_RATIO}）...")
        train_df, val_df = preprocessor.split_dataset(featured_df, train_ratio=config.TRAIN_RATIO)

        # 从DataFrame中提取特征和标签
        # 特征是除了 scfi_next 和 date 之外的所有列
        feature_cols = [col for col in featured_df.columns if col not in ['scfi_next', 'date']]
        X_train = train_df[feature_cols].values
        y_train = train_df['scfi_next'].values
        X_test = val_df[feature_cols].values
        y_test = val_df['scfi_next'].values

        print(f"[OK] 数据预处理完成")
        print(f"  - 训练样本数: {len(X_train)}")
        print(f"  - 测试样本数: {len(X_test)}")
        print(f"  - 特征维度: {X_train.shape[1] if hasattr(X_train, 'shape') else 'N/A'}")

        # ========================
        # 第三步：模型训练
        # ========================
        print_separator("第三步：模型训练")

        # 映射模型类型名称
        model_type_map = {
            'random_forest': 'RandomForest',
            'linear': 'LinearRegression'
        }
        actual_model_type = model_type_map.get(model_type, 'RandomForest')

        # 初始化模型
        model = SCFIForecastModel(
            model_type=actual_model_type,
            model_save_path=str(config.MODEL_SAVE_DIR)
        )

        # 训练模型
        print(f"正在训练 {model_type} 模型...")
        model.train(X_train, y_train)
        print(f"[OK] 模型训练完成")

        # 保存模型
        model_filename = f"{route_name}_{model_type}_model.pkl"
        model.save_model(model_filename)
        print(f"[OK] 模型已保存至: {os.path.join(config.MODEL_SAVE_DIR, model_filename)}")

        # ========================
        # 第四步：模型预测
        # ========================
        print_separator("第四步：模型预测")

        # 在测试集上进行预测
        print("正在进行预测...")
        y_pred = model.predict_next_scfi(X_test)

        # ========================
        # 第五步：结果分析
        # ========================
        print_separator("第五步：结果分析")

        # 初始化结果分析器
        analyzer = ResultAnalyzer()

        # 计算评估指标
        metrics = analyzer.calc_evaluation_metrics(y_test, y_pred)

        print("\n模型评估指标:")
        print("-" * 40)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name:10s}: {metric_value:.6f}")

        # 获取特征重要性
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            # 从随机森林模型获取特征重要性
            feature_importance = dict(zip(feature_cols, model.model.feature_importances_))
        else:
            feature_importance = {}
        if feature_importance:
            print("\n特征重要性 Top 5:")
            print("-" * 40)
            sorted_features = sorted(feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)
            for i, (name, importance) in enumerate(sorted_features[:5], 1):
                print(f"  {i}. {name:30s}: {importance:.6f}")

        # ========================
        # 第六步：结果导出
        # ========================
        print_separator("第六步：结果导出")

        # 初始化结果导出器
        exporter = ResultExporter(output_dir=config.OUTPUT_DIR)

        # 绘制预测对比图
        print("\n正在生成预测对比图...")
        exporter.draw_fit_plot(
            y_true=y_test,
            y_pred=y_pred,
            route_name=route_name
        )

        # 绘制特征重要性图
        if feature_importance:
            print("正在生成特征重要性图...")
            exporter.draw_feature_importance(
                feature_names=list(feature_importance.keys()),
                feature_importance=list(feature_importance.values()),
                route_name=route_name
            )

        # 导出Excel报告
        print("正在导出Excel预测报告...")
        exporter.export_pred_excel(
            y_true=y_test,
            y_pred=y_pred,
            route_name=route_name,
            metrics=metrics,
            feature_importance=feature_importance
        )

        # 导出训练日志
        print("正在导出训练日志...")
        model_info = f"模型类型: {model_type}\n训练参数: 默认配置"
        exporter.export_log(
            model_info=model_info,
            metrics=metrics,
            route_name=route_name,
            feature_importance=feature_importance,
            train_samples=len(X_train),
            test_samples=len(X_test)
        )

        # ========================
        # 完成提示
        # ========================
        print_separator("预测完成")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[OK] 所有结果已保存至: {config.OUTPUT_DIR}")

        # 返回结果字典
        return {
            'route_name': route_name,
            'model_type': model_type,
            'metrics': metrics,
            'y_pred': y_pred,
            'feature_importance': feature_importance
        }

    except FileNotFoundError as e:
        print(f"\n错误: 数据文件未找到 - {e}")
        print("请确保数据文件存在于 data_source 目录下")
        return None

    except Exception as e:
        print(f"\n错误: 程序执行过程中出现异常 - {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    """
    命令行参数解析
    支持参数：
        --route: 选择预测航线
        --model: 选择模型类型
        --help: 显示帮助信息
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='SCFI 海运运价预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用航线选项:
  {'综合指数':<12} - 综合运价指数
  {'欧洲航线':<12} - 欧洲航线运价
  {'地中海航线':<12} - 地中海航线运价
  {'美西航线':<12} - 美国西海岸航线运价
  {'西非航线':<12} - 西非航线运价
  {'南美航线':<12} - 南美航线运价
  {'美东航线':<12} - 美国东海岸航线运价
  {'东南亚航线':<12} - 东南亚航线运价
  {'韩国航线':<12} - 韩国航线运价

示例:
  python main.py --route 综合指数
  python main.py --route 欧洲航线 --model random_forest
  python main.py --route 美西航线 --model xgboost
        """
    )

    # 添加参数选项
    parser.add_argument(
        '--route',
        type=str,
        default='综合指数',
        choices=AVAILABLE_ROUTES,
        help='选择预测航线（默认: 综合指数）'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'linear', 'xgboost'],
        help='选择模型类型（默认: random_forest）'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 执行主函数
    result = main(route_name=args.route, model_type=args.model)

    # 如果执行失败，返回非零退出码
    if result is None:
        sys.exit(1)
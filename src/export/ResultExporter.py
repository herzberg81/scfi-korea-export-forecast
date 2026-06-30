# -*- coding: utf-8 -*-
"""
ResultExporter - 结果导出模块
负责绘制可视化图表、导出Excel预测报告和txt日志文件
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from datetime import datetime

# 设置中文字体支持（带字体检测和 fallback 机制）
def setup_chinese_font():
    """设置中文字体，带有 fallback 机制"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    available_font = None
    
    for font_name in chinese_fonts:
        try:
            # 检查字体是否可用
            font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            for font_path in font_list:
                font_prop = font_manager.FontProperties(fname=font_path)
                if font_name.lower() in font_prop.get_name().lower():
                    available_font = font_name
                    break
            if available_font:
                break
        except:
            continue
    
    if available_font:
        matplotlib.rcParams['font.sans-serif'] = [available_font]
    else:
        # 如果找不到中文字体，使用默认字体（中文可能显示为方块，但程序不会崩溃）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化字体配置
setup_chinese_font()


class ResultExporter:
    """结果导出类，负责可视化图表绘制和数据导出"""

    def __init__(self, output_dir='output'):
        """
        初始化结果导出器

        Args:
            output_dir: 输出目录路径，默认为 'output'
        """
        self.output_dir = output_dir
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def draw_fit_plot(self, y_true, y_pred, route_name='综合指数', save_fig=True):
        """
        绘制真实值与预测值对比折线图

        Args:
            y_true: 真实值列表或数组
            y_pred: 预测值列表或数组
            route_name: 航线名称，默认为'综合指数'
            save_fig: 是否保存图片，默认为True

        Returns:
            fig: matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制真实值和预测值折线
        ax.plot(y_true, label='真实值', color='blue', linewidth=2, marker='o', markersize=4)
        ax.plot(y_pred, label='预测值', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)

        # 设置标题和标签
        ax.set_title(f'SCFI {route_name}运价预测对比图', fontsize=16, fontweight='bold')
        ax.set_xlabel('时间序列（周）', fontsize=12)
        ax.set_ylabel('运价指数', fontsize=12)

        # 添加图例
        ax.legend(loc='best', fontsize=11)

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'SCFI_{route_name}_预测对比_{timestamp}.png'
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 预测对比图已保存至: {save_path}")

        return fig

    def draw_feature_importance(self, feature_names, feature_importance, route_name='综合指数', save_fig=True):
        """
        绘制特征重要性柱状图

        Args:
            feature_names: 特征名称列表
            feature_importance: 特征重要性值列表
            route_name: 航线名称，默认为'综合指数'
            save_fig: 是否保存图片，默认为True

        Returns:
            fig: matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 按重要性排序
        sorted_indices = sorted(range(len(feature_importance)),
                                key=lambda i: feature_importance[i], reverse=True)
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_importance = [feature_importance[i] for i in sorted_indices]

        # 绘制水平柱状图（特征数量多时更易读）
        y_pos = range(len(sorted_names))
        bars = ax.barh(y_pos, sorted_importance, color='steelblue', alpha=0.8)

        # 设置标题和标签
        ax.set_title(f'{route_name}运价预测特征重要性分析', fontsize=16, fontweight='bold')
        ax.set_xlabel('重要性分数', fontsize=12)
        ax.set_ylabel('特征名称', fontsize=12)

        # 设置Y轴刻度标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=10)

        # 在柱状图上显示数值
        for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)

        # 添加网格
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{route_name}_特征重要性_{timestamp}.png'
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 特征重要性图已保存至: {save_path}")

        return fig

    def export_pred_excel(self, y_true, y_pred, dates=None, route_name='综合指数',
                         metrics=None, feature_importance=None):
        """
        导出Excel预测报告

        Args:
            y_true: 真实值列表或数组
            y_pred: 预测值列表或数组
            dates: 日期列表，可选
            route_name: 航线名称
            metrics: 评估指标字典，如 {'R2': 0.95, 'MAE': 10.5, 'RMSE': 15.2}
            feature_importance: 特征重要性字典，如 {'特征名': 重要性值}

        Returns:
            save_path: Excel文件保存路径
        """
        # 创建数据字典
        data_dict = {
            '序号': range(1, len(y_true) + 1),
            '真实值': y_true,
            '预测值': y_pred,
            '预测误差': [true - pred for true, pred in zip(y_true, y_pred)],
            '绝对误差': [abs(true - pred) for true, pred in zip(y_true, y_pred)]
        }

        # 如果提供了日期，添加日期列
        if dates is not None:
            data_dict['日期'] = dates[:len(y_true)]

        # 创建DataFrame（调整列顺序）
        if dates is not None:
            columns_order = ['序号', '日期', '真实值', '预测值', '预测误差', '绝对误差']
        else:
            columns_order = ['序号', '真实值', '预测值', '预测误差', '绝对误差']

        df_pred = pd.DataFrame(data_dict)[columns_order]

        # 生成时间戳文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'SCFI_{route_name}_预测报告_{timestamp}.xlsx'
        save_path = os.path.join(self.output_dir, filename)

        # 使用ExcelWriter写入多个sheet
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # 写入预测结果
            df_pred.to_excel(writer, sheet_name='预测结果', index=False)

            # 写入评估指标（如果提供）
            if metrics is not None:
                df_metrics = pd.DataFrame(list(metrics.items()), columns=['指标名称', '指标值'])
                df_metrics.to_excel(writer, sheet_name='评估指标', index=False)

            # 写入特征重要性（如果提供）
            if feature_importance is not None:
                df_importance = pd.DataFrame(list(feature_importance.items()),
                                            columns=['特征名称', '重要性'])
                df_importance = df_importance.sort_values('重要性', ascending=False)
                df_importance.to_excel(writer, sheet_name='特征重要性', index=False)

        print(f"[OK] Excel预测报告已保存至: {save_path}")
        return save_path

    def export_log(self, model_info, metrics, route_name='综合指数',
                  feature_importance=None, train_samples=0, test_samples=0):
        """
        导出txt日志文件

        Args:
            model_info: 模型信息字符串，如模型类型、参数等
            metrics: 评估指标字典
            route_name: 航线名称
            feature_importance: 特征重要性字典
            train_samples: 训练样本数量
            test_samples: 测试样本数量

        Returns:
            save_path: 日志文件保存路径
        """
        # 生成时间戳文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'SCFI_{route_name}_训练日志_{timestamp}.txt'
        save_path = os.path.join(self.output_dir, filename)

        # 写入日志文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("       SCFI 海运运价预测系统 - 模型训练日志\n")
            f.write("=" * 60 + "\n\n")

            # 写入基本信息
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"预测航线: {route_name}\n")
            f.write(f"训练样本数: {train_samples}\n")
            f.write(f"测试样本数: {test_samples}\n\n")

            # 写入分隔线
            f.write("-" * 60 + "\n")
            f.write("一、模型信息\n")
            f.write("-" * 60 + "\n")
            f.write(f"{model_info}\n\n")

            # 写入评估指标
            f.write("-" * 60 + "\n")
            f.write("二、模型评估指标\n")
            f.write("-" * 60 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.6f}\n")
            f.write("\n")

            # 写入特征重要性
            if feature_importance is not None:
                f.write("-" * 60 + "\n")
                f.write("三、特征重要性排序\n")
                f.write("-" * 60 + "\n")
                sorted_importance = sorted(feature_importance.items(),
                                          key=lambda x: x[1], reverse=True)
                for i, (name, importance) in enumerate(sorted_importance, 1):
                    f.write(f"{i:2d}. {name:30s}: {importance:.6f}\n")
                f.write("\n")

            # 写入结束标识
            f.write("=" * 60 + "\n")
            f.write("                    日志结束\n")
            f.write("=" * 60 + "\n")

        print(f"[OK] 训练日志已保存至: {save_path}")
        return save_path


# 测试代码
if __name__ == '__main__':
    """测试ResultExporter类功能"""
    print("=" * 60)
    print("ResultExporter 功能测试")
    print("=" * 60)

    # 创建导出器实例
    exporter = ResultExporter(output_dir='../../output')

    # 测试数据
    y_true = [1200, 1250, 1180, 1320, 1290, 1350, 1400]
    y_pred = [1195, 1248, 1190, 1310, 1295, 1340, 1395]
    feature_names = ['韩国出口指数_lag1', 'SCFI_lag1', '韩国出口指数_lag2',
                    'SCFI_lag2', '韩国出口指数_lag3', 'SCFI_lag3']
    feature_importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]

    # 测试绘图功能
    print("\n1. 测试绘制预测对比图...")
    fig1 = exporter.draw_fit_plot(y_true, y_pred, route_name='综合指数')

    # 测试特征重要性图
    print("\n2. 测试绘制特征重要性图...")
    fig2 = exporter.draw_feature_importance(feature_names, feature_importance, route_name='综合指数')

    # 测试Excel导出
    print("\n3. 测试导出Excel报告...")
    metrics = {'R2': 0.9856, 'MAE': 8.5, 'RMSE': 10.2}
    feat_dict = dict(zip(feature_names, feature_importance))
    excel_path = exporter.export_pred_excel(y_true, y_pred, route_name='综合指数',
                                            metrics=metrics, feature_importance=feat_dict)

    # 测试日志导出
    print("\n4. 测试导出训练日志...")
    log_path = exporter.export_log(
        model_info="模型类型: RandomForestRegressor\n参数: n_estimators=100, max_depth=10",
        metrics=metrics,
        route_name='综合指数',
        feature_importance=feat_dict,
        train_samples=180,
        test_samples=43
    )

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

    # 关闭图形
    plt.close('all')
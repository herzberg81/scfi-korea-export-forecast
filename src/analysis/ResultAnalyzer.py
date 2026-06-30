"""
SCFI预测结果分析类
提供模型评估指标计算、相关性分析和特征重要性分析功能
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ResultAnalyzer:
    """
    SCFI预测结果分析类
    
    用于计算模型评估指标、相关性分析和特征重要性分析
    """
    
    def __init__(self):
        """
        初始化ResultAnalyzer
        """
        self.metrics = {}
        self.correlation_result = None
        self.feature_importance = None
    
    def calc_evaluation_metrics(self, y_true, y_pred):
        """
        计算模型评估指标：R²、MAE、RMSE
        
        参数:
            y_true: 真实值，形状为 (n_samples,)
            y_pred: 预测值，形状为 (n_samples,)
        
        返回:
            dict: 包含R²、MAE、RMSE的字典
        """
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 计算R²（决定系数），范围[0,1]，越接近1表示模型越好
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAE（平均绝对误差），越小越好
        mae = mean_absolute_error(y_true, y_pred)
        
        # 计算RMSE（均方根误差），越小越好
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 保存结果
        self.metrics = {
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse
        }
        
        # 打印评估结果
        print("=" * 50)
        print("模型评估指标:")
        print("=" * 50)
        print(f"R2 (决定系数): {r2:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print("=" * 50)
        
        return self.metrics
    
    def calc_correlation(self, y_true, y_pred):
        """
        计算真实值与预测值之间的相关系数（Pearson相关系数）
        
        参数:
            y_true: 真实值，形状为 (n_samples,)
            y_pred: 预测值，形状为 (n_samples,)
        
        返回:
            float: Pearson相关系数，范围[-1,1]，越接近1表示正相关越强
        """
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 计算Pearson相关系数
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # 保存结果
        self.correlation_result = correlation
        
        # 打印结果
        print("=" * 50)
        print("相关性分析:")
        print("=" * 50)
        print(f"Pearson相关系数: {correlation:.4f}")
        
        # 解释相关程度
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "高度相关"
        elif abs_corr >= 0.5:
            strength = "中度相关"
        elif abs_corr >= 0.3:
            strength = "弱相关"
        else:
            strength = "极弱相关或无相关"
        
        print(f"相关程度: {strength}")
        print("=" * 50)
        
        return correlation
    
    def get_feature_importance(self, model, feature_names=None):
        """
        获取RandomForest模型的特征重要性
        
        参数:
            model: 训练好的RandomForest模型
            feature_names: 特征名称列表（可选），用于显示特征名称
        
        返回:
            dict: 特征名称和对应重要性分数的字典，按重要性降序排列
        """
        # 检查模型类型
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("该模型不支持特征重要性分析，请使用RandomForest模型")
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 如果没有提供特征名称，使用索引
        if feature_names is None:
            feature_names = [f"特征_{i}" for i in range(len(importances))]
        
        # 创建特征重要性字典
        importance_dict = dict(zip(feature_names, importances))
        
        # 按重要性降序排序
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # 保存结果
        self.feature_importance = sorted_importance
        
        # 打印特征重要性
        print("=" * 50)
        print("特征重要性分析:")
        print("=" * 50)
        for feature, importance in sorted_importance.items():
            print(f"{feature}: {importance:.4f} ({importance*100:.2f}%)")
        print("=" * 50)
        
        return sorted_importance
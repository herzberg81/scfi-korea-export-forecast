"""
SCFI运价预测模型类
支持线性回归和随机森林模型进行海运运价预测
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class SCFIForecastModel:
    """
    SCFI运价预测模型类
    
    属性:
        model_type: 模型类型，支持 'LinearRegression' 和 'RandomForest'
        train_x: 训练特征数据
        train_y: 训练目标数据
        model_save_path: 模型保存路径
        model: 实际的sklearn模型实例
    """
    
    def __init__(self, model_type='LinearRegression', model_save_path='./model'):
        """
        初始化SCFIForecastModel
        
        参数:
            model_type: 模型类型，可选 'LinearRegression' 或 'RandomForest'
            model_save_path: 模型保存的目录路径
        """
        self.model_type = model_type
        self.train_x = None
        self.train_y = None
        self.model_save_path = model_save_path
        self.model = None
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """
        根据model_type初始化对应的sklearn模型
        """
        if self.model_type == 'LinearRegression':
            self.model = LinearRegression()
        elif self.model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}，请选择 'LinearRegression' 或 'RandomForest'")
    
    def train(self, train_x, train_y):
        """
        使用训练数据训练模型
        
        参数:
            train_x: 训练特征数据，形状为 (n_samples, n_features)
            train_y: 训练目标数据，形状为 (n_samples,)
        
        返回:
            self: 返回模型实例，支持链式调用
        """
        # 保存训练数据
        self.train_x = train_x
        self.train_y = train_y
        
        # 训练模型
        self.model.fit(train_x, train_y)
        
        print(f"模型训练完成！模型类型: {self.model_type}")
        return self
    
    def save_model(self, filename='scfi_model.pkl'):
        """
        使用joblib保存训练好的模型到文件
        
        参数:
            filename: 模型保存的文件名，默认为 'scfi_model.pkl'
        
        返回:
            str: 模型保存的完整路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法训练模型")
        
        # 创建保存目录（如果不存在）
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        # 构建完整保存路径
        save_path = os.path.join(self.model_save_path, filename)
        
        # 使用joblib保存模型
        joblib.dump(self.model, save_path)
        
        print(f"模型已保存至: {save_path}")
        return save_path
    
    def load_model(self, filepath):
        """
        从文件加载已保存的模型
        
        参数:
            filepath: 模型文件的完整路径
        
        返回:
            self: 返回模型实例，支持链式调用
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 使用joblib加载模型
        self.model = joblib.load(filepath)
        
        # 根据加载的模型类型更新model_type
        if isinstance(self.model, LinearRegression):
            self.model_type = 'LinearRegression'
        elif isinstance(self.model, RandomForestRegressor):
            self.model_type = 'RandomForest'
        else:
            self.model_type = type(self.model).__name__
        
        print(f"模型已加载: {filepath}")
        print(f"模型类型: {self.model_type}")
        return self
    
    def predict_next_scfi(self, features):
        """
        预测下周SCFI综合运价指数
        
        参数:
            features: 预测特征数据，形状为 (n_samples, n_features)
        
        返回:
            numpy.ndarray: 预测的SCFI运价值，形状为 (n_samples,)
        """
        if self.model is None:
            raise ValueError("模型尚未加载或训练，请先调用train()或load_model()方法")
        
        # 将输入转换为numpy数组（如果是list或其他格式）
        features = np.array(features)
        
        # 如果特征是一维的，转换为二维
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 进行预测
        prediction = self.model.predict(features)
        
        return prediction
    
    def predict_route(self, route_features, route_name=None):
        """
        预测特定航线的SCFI运价
        
        参数:
            route_features: 该航线的特征数据，形状为 (n_samples, n_features)
            route_name: 航线名称（可选），用于输出显示
        
        返回:
            numpy.ndarray: 预测的航线运价值，形状为 (n_samples,)
        """
        if self.model is None:
            raise ValueError("模型尚未加载或训练，请先调用train()或load_model()方法")
        
        # 将输入转换为numpy数组
        route_features = np.array(route_features)
        
        # 如果特征是一维的，转换为二维
        if route_features.ndim == 1:
            route_features = route_features.reshape(1, -1)
        
        # 进行预测
        prediction = self.model.predict(route_features)
        
        # 显示预测结果
        if route_name:
            print(f"航线 [{route_name}] 预测结果: {prediction}")
        
        return prediction
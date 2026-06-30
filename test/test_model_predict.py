"""
SCFIForecastModel和ResultAnalyzer类的单元测试
测试模型训练、预测和结果分析功能
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.SCFIForecastModel import SCFIForecastModel
from src.analysis.ResultAnalyzer import ResultAnalyzer


class TestSCFIForecastModel(unittest.TestCase):
    """SCFIForecastModel类的测试用例"""

    def setUp(self):
        """测试前的准备工作
        创建训练数据和临时目录
        """
        # 设置随机种子以保证结果可复现
        np.random.seed(42)

        # 创建模拟训练数据
        self.n_samples = 100
        self.n_features = 5

        # 生成特征数据（包含滞后特征）
        self.train_x = np.random.randn(self.n_samples, self.n_features)

        # 生成目标数据（与特征有线性关系）
        self.train_y = (self.train_x[:, 0] * 2 +
                       self.train_x[:, 1] * 1.5 +
                       self.train_x[:, 2] * 0.5 +
                       np.random.randn(self.n_samples) * 0.1)

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_linear_regression(self):
        """测试初始化线性回归模型"""
        model = SCFIForecastModel(model_type='LinearRegression')

        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, 'LinearRegression')
        self.assertIsInstance(model.model, LinearRegression)
        print("[OK] 测试初始化线性回归模型成功")

    def test_init_random_forest(self):
        """测试初始化随机森林模型"""
        model = SCFIForecastModel(model_type='RandomForest')

        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, 'RandomForest')
        self.assertIsInstance(model.model, RandomForestRegressor)
        print("[OK] 测试初始化随机森林模型成功")

    def test_init_invalid_model_type(self):
        """测试初始化不支持的模型类型"""
        with self.assertRaises(ValueError):
            SCFIForecastModel(model_type='InvalidModel')
        print("[OK] 测试无效模型类型异常处理成功")

    def test_train(self):
        """测试模型训练"""
        model = SCFIForecastModel(model_type='LinearRegression')

        # 训练模型
        trained_model = model.train(self.train_x, self.train_y)

        # 验证训练成功
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(model.model)
        self.assertTrue(hasattr(model.model, 'coef_'))  # 线性回归模型应该有coef_属性
        print("[OK] 测试模型训练成功")

    def test_train_random_forest(self):
        """测试随机森林模型训练"""
        model = SCFIForecastModel(model_type='RandomForest')

        # 训练模型
        trained_model = model.train(self.train_x, self.train_y)

        # 验证训练成功
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(model.model)
        print("[OK] 测试随机森林模型训练成功")

    def test_predict_next_scfi(self):
        """测试预测下周SCFI运价"""
        # 训练模型
        model = SCFIForecastModel(model_type='LinearRegression')
        model.train(self.train_x, self.train_y)

        # 预测
        predictions = model.predict_next_scfi(self.train_x)

        # 验证预测结果
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), self.n_samples)
        print(f"[OK] 测试预测下周SCFI运价成功，预测了 {len(predictions)} 个值")

    def test_predict_with_1d_features(self):
        """测试使用一维特征进行预测"""
        # 训练模型
        model = SCFIForecastModel(model_type='LinearRegression')
        model.train(self.train_x, self.train_y)

        # 使用一维特征预测
        single_feature = self.train_x[0]
        prediction = model.predict_next_scfi(single_feature)

        # 验证预测结果
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 1)
        print("[OK] 测试一维特征预测成功")

    def test_predict_route(self):
        """测试预测特定航线运价"""
        # 训练模型
        model = SCFIForecastModel(model_type='LinearRegression')
        model.train(self.train_x, self.train_y)

        # 预测特定航线
        route_features = self.train_x[:5]
        predictions = model.predict_route(route_features, route_name="测试航线")

        # 验证预测结果
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 5)
        print("[OK] 测试预测特定航线运价成功")

    def test_predict_without_training(self):
        """测试未训练模型时的预测异常"""
        model = SCFIForecastModel(model_type='LinearRegression')

        # 未训练直接预测应该抛出异常
        with self.assertRaises(ValueError):
            model.predict_next_scfi(self.train_x)
        print("[OK] 测试未训练模型预测异常处理成功")

    def test_save_model(self):
        """测试保存模型"""
        # 训练模型
        model = SCFIForecastModel(
            model_type='LinearRegression',
            model_save_path=self.temp_dir
        )
        model.train(self.train_x, self.train_y)

        # 保存模型
        saved_path = model.save_model(filename='test_model.pkl')

        # 验证文件已创建
        self.assertTrue(os.path.exists(saved_path))
        print(f"[OK] 测试保存模型成功，保存至: {saved_path}")

    def test_save_model_without_training(self):
        """测试未训练时保存模型的异常"""
        model = SCFIForecastModel(
            model_type='LinearRegression',
            model_save_path=self.temp_dir
        )

        # 手动设置model为None，模拟未训练状态
        # 因为_init_model会在初始化时创建模型实例，所以我们需要手动设置为None
        model.model = None

        # 未训练时保存应该抛出异常
        with self.assertRaises(ValueError):
            model.save_model(filename='test_model.pkl')
        print("[OK] 测试未训练模型保存异常处理成功")

    def test_load_model(self):
        """测试加载模型"""
        # 训练并保存模型
        model1 = SCFIForecastModel(
            model_type='LinearRegression',
            model_save_path=self.temp_dir
        )
        model1.train(self.train_x, self.train_y)
        saved_path = model1.save_model(filename='test_model.pkl')

        # 加载模型
        model2 = SCFIForecastModel(model_type='LinearRegression')
        model2.load_model(saved_path)

        # 验证加载成功
        self.assertIsNotNone(model2.model)
        self.assertEqual(model2.model_type, 'LinearRegression')

        # 验证预测结果一致
        pred1 = model1.predict_next_scfi(self.train_x[:5])
        pred2 = model2.predict_next_scfi(self.train_x[:5])
        np.testing.assert_array_almost_equal(pred1, pred2)
        print("[OK] 测试加载模型成功，预测结果一致")

    def test_load_model_file_not_found(self):
        """测试加载不存在的模型文件"""
        model = SCFIForecastModel(model_type='LinearRegression')

        # 加载不存在的文件应该抛出异常
        with self.assertRaises(FileNotFoundError):
            model.load_model('nonexistent_model.pkl')
        print("[OK] 测试加载不存在模型文件异常处理成功")

    def test_random_forest_feature_importance(self):
        """测试随机森林模型的特征重要性"""
        # 训练随机森林模型
        model = SCFIForecastModel(model_type='RandomForest')
        model.train(self.train_x, self.train_y)

        # 验证模型有特征重要性属性
        self.assertTrue(hasattr(model.model, 'feature_importances_'))
        self.assertEqual(len(model.model.feature_importances_), self.n_features)
        print("[OK] 测试随机森林特征重要性成功")


class TestResultAnalyzer(unittest.TestCase):
    """ResultAnalyzer类的测试用例"""

    def setUp(self):
        """测试前的准备工作
        创建预测结果数据
        """
        # 设置随机种子
        np.random.seed(42)

        # 创建真实值和预测值
        self.y_true = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        # 预测值与真实值有较好的相关性
        self.y_pred = self.y_true + np.random.randn(10) * 0.5

        # 创建分析器实例
        self.analyzer = ResultAnalyzer()

    def test_init(self):
        """测试初始化方法"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.metrics, {})
        self.assertIsNone(self.analyzer.correlation_result)
        self.assertIsNone(self.analyzer.feature_importance)
        print("[OK] 测试ResultAnalyzer初始化成功")

    def test_calc_evaluation_metrics(self):
        """测试计算评估指标"""
        # 计算评估指标
        metrics = self.analyzer.calc_evaluation_metrics(self.y_true, self.y_pred)

        # 验证返回的指标
        self.assertIn('R2', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)

        # 验证指标值合理
        self.assertGreaterEqual(metrics['R2'], -1)  # R2可以小于0（模型很差时）
        self.assertGreaterEqual(metrics['MAE'], 0)
        self.assertGreaterEqual(metrics['RMSE'], 0)

        print(f"[OK] 测试计算评估指标成功: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    def test_calc_evaluation_metrics_perfect_prediction(self):
        """测试完美预测的评估指标"""
        # 完美预测（预测值=真实值）
        y_pred_perfect = self.y_true.copy()

        metrics = self.analyzer.calc_evaluation_metrics(self.y_true, y_pred_perfect)

        # 验证完美预测的指标
        self.assertAlmostEqual(metrics['R2'], 1.0, places=5)
        self.assertAlmostEqual(metrics['MAE'], 0.0, places=5)
        self.assertAlmostEqual(metrics['RMSE'], 0.0, places=5)

        print("[OK] 测试完美预测评估指标成功")

    def test_calc_correlation(self):
        """测试计算相关系数"""
        # 计算相关系数
        correlation = self.analyzer.calc_correlation(self.y_true, self.y_pred)

        # 验证相关系数
        self.assertIsNotNone(correlation)
        self.assertGreaterEqual(correlation, -1)
        self.assertLessEqual(correlation, 1)
        self.assertEqual(self.analyzer.correlation_result, correlation)

        print(f"[OK] 测试计算相关系数成功: 相关系数={correlation:.4f}")

    def test_calc_correlation_perfect(self):
        """测试完美相关的相关系数"""
        # 完美相关
        y_pred_perfect = self.y_true * 2 + 10  # 完全线性相关

        correlation = self.analyzer.calc_correlation(self.y_true, y_pred_perfect)

        # 验证相关系数接近1
        self.assertAlmostEqual(correlation, 1.0, places=5)
        print("[OK] 测试完美相关系数成功")

    def test_calc_correlation_negative(self):
        """测试负相关的相关系数"""
        # 负相关
        y_pred_negative = -self.y_true

        correlation = self.analyzer.calc_correlation(self.y_true, y_pred_negative)

        # 验证相关系数接近-1
        self.assertAlmostEqual(correlation, -1.0, places=5)
        print("[OK] 测试负相关系数成功")

    def test_get_feature_importance(self):
        """测试获取特征重要性"""
        # 创建随机森林模型并训练
        np.random.seed(42)
        n_samples = 50
        n_features = 5
        train_x = np.random.randn(n_samples, n_features)
        train_y = (train_x[:, 0] * 2 + train_x[:, 1] * 1.5 +
                  np.random.randn(n_samples) * 0.1)

        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(train_x, train_y)

        # 获取特征重要性
        feature_names = ['特征1', '特征2', '特征3', '特征4', '特征5']
        importance = self.analyzer.get_feature_importance(rf_model, feature_names)

        # 验证返回的特征重要性
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), n_features)
        self.assertIsInstance(importance, dict)

        # 验证重要性之和接近1
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=2)

        print(f"[OK] 测试获取特征重要性成功，共 {len(importance)} 个特征")

    def test_get_feature_importance_without_names(self):
        """测试不提供特征名称时获取特征重要性"""
        # 创建随机森林模型并训练
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        train_x = np.random.randn(n_samples, n_features)
        train_y = train_x[:, 0] * 2 + np.random.randn(n_samples) * 0.1

        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(train_x, train_y)

        # 不提供特征名称
        importance = self.analyzer.get_feature_importance(rf_model)

        # 验证使用了默认特征名称
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), n_features)
        self.assertTrue(all('特征_' in name for name in importance.keys()))

        print("[OK] 测试不提供特征名称获取特征重要性成功")

    def test_get_feature_importance_invalid_model(self):
        """测试对不支持特征重要性的模型获取特征重要性"""
        # 创建线性回归模型（不支持特征重要性）
        lr_model = LinearRegression()
        lr_model.fit(np.random.randn(10, 3), np.random.randn(10))

        # 应该抛出异常
        with self.assertRaises(ValueError):
            self.analyzer.get_feature_importance(lr_model)
        print("[OK] 测试无效模型特征重要性异常处理成功")


class TestModelAndAnalyzerIntegration(unittest.TestCase):
    """SCFIForecastModel和ResultAnalyzer的集成测试"""

    def setUp(self):
        """测试前的准备工作"""
        # 设置随机种子
        np.random.seed(42)

        # 创建模拟数据
        self.n_samples = 100
        self.n_features = 5

        # 训练数据
        self.train_x = np.random.randn(80, self.n_features)
        self.train_y = (self.train_x[:, 0] * 2 +
                       self.train_x[:, 1] * 1.5 +
                       np.random.randn(80) * 0.1)

        # 测试数据
        self.test_x = np.random.randn(20, self.n_features)
        self.test_y = (self.test_x[:, 0] * 2 +
                      self.test_x[:, 1] * 1.5 +
                      np.random.randn(20) * 0.1)

    def test_complete_workflow(self):
        """测试完整的模型训练、预测和分析流程"""
        # 1. 训练模型
        model = SCFIForecastModel(model_type='LinearRegression')
        model.train(self.train_x, self.train_y)
        print("步骤1: 模型训练完成")

        # 2. 进行预测
        predictions = model.predict_next_scfi(self.test_x)
        print(f"步骤2: 完成 {len(predictions)} 个预测")

        # 3. 分析结果
        analyzer = ResultAnalyzer()
        metrics = analyzer.calc_evaluation_metrics(self.test_y, predictions)
        correlation = analyzer.calc_correlation(self.test_y, predictions)

        print(f"步骤3: 结果分析完成")
        print(f"  - R2: {metrics['R2']:.4f}")
        print(f"  - MAE: {metrics['MAE']:.4f}")
        print(f"  - RMSE: {metrics['RMSE']:.4f}")
        print(f"  - 相关系数: {correlation:.4f}")

        # 验证整个流程
        self.assertIsNotNone(predictions)
        self.assertIn('R2', metrics)
        self.assertIsNotNone(correlation)

        print("[OK] 测试完整工作流程成功")

    def test_random_forest_with_feature_importance(self):
        """测试随机森林模型的完整流程包括特征重要性分析"""
        # 1. 训练随机森林模型
        model = SCFIForecastModel(model_type='RandomForest')
        model.train(self.train_x, self.train_y)
        print("步骤1: 随机森林模型训练完成")

        # 2. 进行预测
        predictions = model.predict_next_scfi(self.test_x)
        print(f"步骤2: 完成 {len(predictions)} 个预测")

        # 3. 分析结果
        analyzer = ResultAnalyzer()
        metrics = analyzer.calc_evaluation_metrics(self.test_y, predictions)

        # 4. 获取特征重要性
        feature_names = [f'特征{i+1}' for i in range(self.n_features)]
        importance = analyzer.get_feature_importance(model.model, feature_names)

        print(f"步骤3: 特征重要性分析完成")
        print("  特征重要性排序:")
        for i, (feature, score) in enumerate(list(importance.items())[:3]):
            print(f"    {i+1}. {feature}: {score:.4f}")

        # 验证
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), self.n_features)

        print("[OK] 测试随机森林完整流程成功")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
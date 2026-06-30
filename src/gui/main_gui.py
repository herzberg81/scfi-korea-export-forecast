# -*- coding: utf-8 -*-
"""
海运运价预测系统 - GUI界面
基于Tkinter构建的图形用户界面，方便用户操作预测系统
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.config.train_config import TrainConfig
from src.data.DataReader import DataReader
from src.data.DataFusionPreprocessor import DataFusionPreprocessor
from src.model.SCFIForecastModel import SCFIForecastModel
from src.analysis.ResultAnalyzer import ResultAnalyzer
from src.export.ResultExporter import ResultExporter


class SCFIGUI:
    """海运运价预测系统GUI类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SCFI 海运运价预测系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # 设置全局样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 初始化变量
        self.selected_route = tk.StringVar(value='综合指数')
        self.selected_model = tk.StringVar(value='RandomForest')
        self.is_running = False
        self.y_true = None
        self.y_pred = None
        self.metrics = None
        self.importance = None
        self.feature_cols = None
        
        # 创建主布局
        self.create_widgets()
    
    def create_widgets(self):
        """创建所有界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部标题区域
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="📦 SCFI 海运运价预测系统", 
                                font=('Microsoft YaHei', 18, 'bold'))
        title_label.pack(pady=10)
        
        subtitle_label = ttk.Label(title_frame, 
                                   text="基于韩国出口先行指数 + SCFI 上海集装箱运价指数的周度运价预测",
                                   font=('Microsoft YaHei', 10), foreground='gray')
        subtitle_label.pack()
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 航线选择
        route_label = ttk.Label(control_frame, text="预测航线:", font=('Microsoft YaHei', 10))
        route_label.pack(anchor=tk.W, pady=(5, 2))
        
        routes = ['综合指数', '欧洲航线', '地中海航线', '美西航线', '西非航线',
                  '南美航线', '美东航线', '东南亚航线', '韩国航线']
        route_combo = ttk.Combobox(control_frame, textvariable=self.selected_route, 
                                    values=routes, state='readonly', width=20)
        route_combo.pack(fill=tk.X, pady=(0, 10))
        
        # 模型选择
        model_label = ttk.Label(control_frame, text="模型类型:", font=('Microsoft YaHei', 10))
        model_label.pack(anchor=tk.W, pady=(5, 2))
        
        models = [('随机森林', 'RandomForest'), ('线性回归', 'LinearRegression')]
        for text, value in models:
            rb = ttk.Radiobutton(control_frame, text=text, variable=self.selected_model, value=value)
            rb.pack(anchor=tk.W)
        
        # 运行按钮
        run_btn = ttk.Button(control_frame, text="🚀 开始预测", command=self.run_prediction,
                              style='Accent.TButton')
        run_btn.pack(fill=tk.X, pady=(20, 5))
        
        clear_btn = ttk.Button(control_frame, text="🔄 清空结果", command=self.clear_results)
        clear_btn.pack(fill=tk.X, pady=(0, 5))
        
        open_folder_btn = ttk.Button(control_frame, text="📂 打开输出目录", command=self.open_output_folder)
        open_folder_btn.pack(fill=tk.X)
        
        # 数据统计信息
        stats_frame = ttk.LabelFrame(control_frame, text="数据统计", padding="10")
        stats_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.scfi_count_label = ttk.Label(stats_frame, text="SCFI数据: -- 条")
        self.scfi_count_label.pack(anchor=tk.W)
        
        self.korea_count_label = ttk.Label(stats_frame, text="韩国出口指数: -- 条")
        self.korea_count_label.pack(anchor=tk.W)
        
        self.train_count_label = ttk.Label(stats_frame, text="训练样本: --")
        self.train_count_label.pack(anchor=tk.W)
        
        self.test_count_label = ttk.Label(stats_frame, text="测试样本: --")
        self.test_count_label.pack(anchor=tk.W)
        
        self.feature_count_label = ttk.Label(stats_frame, text="特征维度: --")
        self.feature_count_label.pack(anchor=tk.W)
        
        # 右侧主显示区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 选项卡
        tab_control = ttk.Notebook(right_frame)
        
        # 结果选项卡
        result_tab = ttk.Frame(tab_control)
        tab_control.add(result_tab, text='预测结果')
        
        # 评估指标
        metrics_frame = ttk.LabelFrame(result_tab, text="模型评估指标", padding="10")
        metrics_frame.pack(fill=tk.X, pady=(5, 10))
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        ttk.Label(metrics_grid, text="R² (决定系数):", font=('Microsoft YaHei', 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.r2_label = ttk.Label(metrics_grid, text="--", font=('Microsoft YaHei', 10, 'bold'), foreground='#2E7D32')
        self.r2_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(metrics_grid, text="MAE (平均绝对误差):", font=('Microsoft YaHei', 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.mae_label = ttk.Label(metrics_grid, text="--", font=('Microsoft YaHei', 10, 'bold'), foreground='#1565C0')
        self.mae_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(metrics_grid, text="RMSE (均方根误差):", font=('Microsoft YaHei', 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.rmse_label = ttk.Label(metrics_grid, text="--", font=('Microsoft YaHei', 10, 'bold'), foreground='#C62828')
        self.rmse_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        
        # 特征重要性
        importance_frame = ttk.LabelFrame(result_tab, text="特征重要性 Top 10", padding="10")
        importance_frame.pack(fill=tk.BOTH, expand=True)
        
        self.importance_tree = ttk.Treeview(importance_frame, columns=('rank', 'feature', 'importance'), show='headings')
        self.importance_tree.heading('rank', text='排名')
        self.importance_tree.heading('feature', text='特征名称')
        self.importance_tree.heading('importance', text='重要性')
        self.importance_tree.column('rank', width=50, anchor=tk.CENTER)
        self.importance_tree.column('feature', width=200, anchor=tk.W)
        self.importance_tree.column('importance', width=100, anchor=tk.CENTER)
        self.importance_tree.pack(fill=tk.BOTH, expand=True)
        
        # 图表选项卡
        chart_tab = ttk.Frame(tab_control)
        tab_control.add(chart_tab, text='可视化图表')
        
        chart_notebook = ttk.Notebook(chart_tab)
        
        # 预测对比图
        fit_plot_frame = ttk.Frame(chart_notebook)
        chart_notebook.add(fit_plot_frame, text='预测对比')
        self.fit_plot_canvas = None
        
        # 特征重要性图
        importance_plot_frame = ttk.Frame(chart_notebook)
        chart_notebook.add(importance_plot_frame, text='特征重要性')
        self.importance_plot_canvas = None
        
        chart_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 日志选项卡
        log_tab = ttk.Frame(tab_control)
        tab_control.add(log_tab, text='运行日志')
        
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # 底部状态条
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        # 加载数据统计
        self.load_data_stats()
    
    def load_data_stats(self):
        """加载数据统计信息"""
        try:
            scfi_file = TrainConfig.SCFI_DATA_FILE
            korea_file = TrainConfig.KOREA_EXPORT_DATA_FILE
            
            if os.path.exists(scfi_file):
                scfi_df = pd.read_excel(scfi_file, header=0)
                self.scfi_count_label.config(text=f"SCFI数据: {len(scfi_df)} 条")
            
            if os.path.exists(korea_file):
                korea_df = pd.read_excel(korea_file, skiprows=1, header=0)
                self.korea_count_label.config(text=f"韩国出口指数: {len(korea_df)} 条")
                
        except Exception as e:
            self.log(f"加载数据统计失败: {str(e)}")
    
    def log(self, message):
        """在日志区域添加消息"""
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_prediction(self):
        """运行预测流程（在后台线程中执行）"""
        if self.is_running:
            messagebox.showwarning("提示", "预测正在进行中，请等待完成")
            return
        
        self.is_running = True
        self.status_var.set("正在预测...")
        
        # 在后台线程中执行预测
        thread = threading.Thread(target=self._run_prediction_thread)
        thread.start()
    
    def _run_prediction_thread(self):
        """预测线程执行函数"""
        try:
            route = self.selected_route.get()
            model_type = self.selected_model.get()
            
            self.log(f"========== 开始预测 ==========")
            self.log(f"预测航线: {route}")
            self.log(f"模型类型: {model_type}")
            
            # 步骤1：读取数据
            self.log("\n[步骤1] 读取数据...")
            reader = DataReader(scfi_file=TrainConfig.SCFI_DATA_FILE,
                               korea_export_file=TrainConfig.KOREA_EXPORT_DATA_FILE)
            scfi_df, korea_df = reader.get_original_two_index()
            self.log(f"SCFI数据读取成功，共 {len(scfi_df)} 条")
            self.log(f"韩国出口指数读取成功，共 {len(korea_df)} 条")
            
            # 步骤2：数据预处理与融合
            self.log("\n[步骤2] 数据预处理与融合...")
            preprocessor = DataFusionPreprocessor(lag_weeks=TrainConfig.LAG_WEEKS,
                                                 train_ratio=TrainConfig.TRAIN_RATIO)
            fusion_df = preprocessor.merge_two_index(scfi_df, korea_df)
            cleaned_df = preprocessor.clean_data(fusion_df)
            featured_df = preprocessor.build_lag_feature(cleaned_df, target_col=route)
            train_df, val_df = preprocessor.split_dataset(featured_df)
            
            # 获取特征列
            self.feature_cols = [col for col in featured_df.columns 
                               if col != 'date' and col != 'scfi_next' and col != route]
            X_train = train_df[self.feature_cols].values
            y_train = train_df['scfi_next'].values
            X_val = val_df[self.feature_cols].values
            y_val = val_df['scfi_next'].values
            
            # 更新统计信息
            self.train_count_label.config(text=f"训练样本: {len(train_df)}")
            self.test_count_label.config(text=f"测试样本: {len(val_df)}")
            self.feature_count_label.config(text=f"特征维度: {len(self.feature_cols)}")
            
            self.log(f"数据预处理完成")
            self.log(f"训练样本数: {len(train_df)}")
            self.log(f"测试样本数: {len(val_df)}")
            self.log(f"特征维度: {len(self.feature_cols)}")
            
            # 步骤3：模型训练
            self.log("\n[步骤3] 模型训练...")
            model = SCFIForecastModel(model_type=model_type)
            model.train(X_train, y_train)
            model.save_model()
            self.log(f"模型训练完成，已保存")
            
            # 步骤4：模型预测
            self.log("\n[步骤4] 模型预测...")
            y_pred = model.predict_next_scfi(X_val)
            self.y_true = y_val
            self.y_pred = y_pred
            self.log(f"预测完成，共 {len(y_pred)} 个预测值")
            
            # 步骤5：结果分析
            self.log("\n[步骤5] 结果分析...")
            analyzer = ResultAnalyzer()
            self.metrics = analyzer.calc_evaluation_metrics(y_val, y_pred)
            
            # 获取特征重要性
            try:
                self.importance = analyzer.get_feature_importance(model.model, self.feature_cols)
            except ValueError:
                self.importance = None
            
            self.log(f"评估指标计算完成")
            self.log(f"R²: {self.metrics['R2']:.4f}")
            self.log(f"MAE: {self.metrics['MAE']:.4f}")
            self.log(f"RMSE: {self.metrics['RMSE']:.4f}")
            
            # 步骤6：结果导出
            self.log("\n[步骤6] 结果导出...")
            exporter = ResultExporter()
            exporter.draw_fit_plot(y_val, y_pred)
            if self.importance:
                exporter.draw_feature_importance(
                    feature_names=list(self.importance.keys()),
                    feature_importance=list(self.importance.values()),
                    route_name=route
                )
            exporter.export_pred_excel(y_val, y_pred, route_name=route, metrics=self.metrics)
            exporter.export_log(model_info=model_type, metrics=self.metrics, 
                               route_name=route, feature_importance=self.importance)
            self.log(f"结果导出完成")
            
            # 更新GUI显示
            self.root.after(0, self.update_results_display)
            
            self.log("\n========== 预测完成 ==========")
            self.status_var.set("预测完成")
            
        except Exception as e:
            self.log(f"❌ 预测失败: {str(e)}")
            self.status_var.set("预测失败")
            messagebox.showerror("错误", f"预测失败: {str(e)}")
        
        finally:
            self.is_running = False
    
    def update_results_display(self):
        """更新结果显示"""
        # 更新评估指标
        if self.metrics:
            self.r2_label.config(text=f"{self.metrics['R2']:.4f}")
            self.mae_label.config(text=f"{self.metrics['MAE']:.4f}")
            self.rmse_label.config(text=f"{self.metrics['RMSE']:.4f}")
        
        # 更新特征重要性
        if self.importance:
            # 清空现有数据
            for item in self.importance_tree.get_children():
                self.importance_tree.delete(item)
            
            # 添加前10个特征
            for i, (feature, score) in enumerate(list(self.importance.items())[:10], 1):
                self.importance_tree.insert('', tk.END, values=(i, feature, f"{score:.4f}"))
        
        # 绘制图表
        self.plot_fit_plot()
        if self.importance:
            self.plot_importance_plot()
    
    def plot_fit_plot(self):
        """绘制预测对比图"""
        if self.y_true is None or self.y_pred is None:
            return
        
        # 创建画布区域
        fit_plot_frame = self.root.winfo_children()[0].winfo_children()[2].winfo_children()[0].winfo_children()[1].winfo_children()[0].winfo_children()[0]
        
        # 清除旧画布
        for widget in fit_plot_frame.winfo_children():
            widget.destroy()
        
        # 创建新图表
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        ax.plot(self.y_true, 'b-', label='真实值', linewidth=1.5)
        ax.plot(self.y_pred, 'r--', label='预测值', linewidth=1.5)
        ax.set_title(f'{self.selected_route.get()} 预测对比图', fontsize=12)
        ax.set_xlabel('时间（周）', fontsize=10)
        ax.set_ylabel('SCFI指数', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, master=fit_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fit_plot_canvas = canvas
    
    def plot_importance_plot(self):
        """绘制特征重要性图"""
        if self.importance is None:
            return
        
        # 创建画布区域
        importance_plot_frame = self.root.winfo_children()[0].winfo_children()[2].winfo_children()[0].winfo_children()[1].winfo_children()[0].winfo_children()[1]
        
        # 清除旧画布
        for widget in importance_plot_frame.winfo_children():
            widget.destroy()
        
        # 获取前10个重要特征
        top_features = list(self.importance.items())[:10]
        features = [f[0] for f in top_features]
        scores = [f[1] for f in top_features]
        
        # 创建新图表
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores, color='#42A5F5')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('重要性', fontsize=10)
        ax.set_title('特征重要性排序', fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        
        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, master=importance_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.importance_plot_canvas = canvas
    
    def clear_results(self):
        """清空结果"""
        self.y_true = None
        self.y_pred = None
        self.metrics = None
        self.importance = None
        
        # 清空指标
        self.r2_label.config(text="--")
        self.mae_label.config(text="--")
        self.rmse_label.config(text="--")
        
        # 清空特征重要性
        for item in self.importance_tree.get_children():
            self.importance_tree.delete(item)
        
        # 清空图表
        fit_plot_frame = self.root.winfo_children()[0].winfo_children()[2].winfo_children()[0].winfo_children()[1].winfo_children()[0].winfo_children()[0]
        for widget in fit_plot_frame.winfo_children():
            widget.destroy()
        
        importance_plot_frame = self.root.winfo_children()[0].winfo_children()[2].winfo_children()[0].winfo_children()[1].winfo_children()[0].winfo_children()[1]
        for widget in importance_plot_frame.winfo_children():
            widget.destroy()
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        
        # 重置统计
        self.train_count_label.config(text="训练样本: --")
        self.test_count_label.config(text="测试样本: --")
        self.feature_count_label.config(text="特征维度: --")
        
        self.status_var.set("已清空")
    
    def open_output_folder(self):
        """打开输出目录"""
        output_dir = TrainConfig.OUTPUT_DIR
        if os.path.exists(output_dir):
            os.startfile(output_dir)
        else:
            messagebox.showwarning("提示", "输出目录不存在")


if __name__ == '__main__':
    from datetime import datetime
    
    root = tk.Tk()
    app = SCFIGUI(root)
    root.mainloop()
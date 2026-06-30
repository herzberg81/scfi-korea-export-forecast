# SCFI韩国出口运价预测软件

## 项目简介

本项目是一个基于机器学习的海运运价预测系统，主要针对上海出口集装箱运价指数（SCFI）中的韩国航线进行运价预测。系统通过融合韩国出口指数和上海出口集装箱运价指数的历史数据，构建预测模型，为海运物流决策提供数据支持。

## 功能特点

- **数据融合处理**：整合多个数据源，进行数据清洗和预处理
- **机器学习预测**：基于历史数据训练预测模型
- **结果分析**：提供预测结果的可视化分析
- **结果导出**：支持预测结果的导出功能

## 目录结构

```
scfi_korea_export_forecast/
├── data_source/              # 数据源目录
│   ├── 上海出口集装箱运价指数.xlsx
│   └── 韩国出口指数.xlsx
├── docs/                     # 项目文档
│   ├── 需求分析文档.md
│   ├── 总体设计文档.md
│   ├── 详细设计文档.md
│   ├── 用户操作手册.md
│   └── 测试报告.md
├── src/                      # 源代码目录
│   ├── analysis/             # 分析模块
│   │   └── ResultAnalyzer.py
│   ├── config/               # 配置模块
│   │   └── train_config.py
│   ├── data/                 # 数据处理模块
│   │   ├── DataReader.py
│   │   └── DataFusionPreprocessor.py
│   ├── entity/               # 实体定义
│   │   └── entity.py
│   ├── export/               # 导出模块
│   │   └── ResultExporter.py
│   ├── model/                # 模型模块
│   │   └── SCFIForecastModel.py
│   └── main.py               # 主程序入口
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略配置
└── README.md                # 项目说明文档
```

## 安装步骤

### 1. 环境要求

- Python 3.8+
- pip 包管理器

### 2. 克隆项目

```bash
git clone <项目仓库地址>
cd scfi_korea_export_forecast
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行预测程序

```bash
python src/main.py
```

### 配置说明

可在 `src/config/train_config.py` 中修改模型训练参数。

## 数据说明

### 数据来源

- **上海出口集装箱运价指数.xlsx**：包含上海出口集装箱运价指数（SCFI）的历史数据
- **韩国出口指数.xlsx**：包含韩国出口贸易指数数据

### 数据格式

数据文件采用 Excel 格式（.xlsx），包含时间序列数据，用于模型训练和预测分析。

## 技术栈

- **数据处理**：pandas、numpy
- **机器学习**：scikit-learn、statsmodels
- **可视化**：matplotlib
- **模型持久化**：joblib
- **Excel读取**：openpyxl

## 许可证

本项目仅供学习和研究使用。
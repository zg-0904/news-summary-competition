\# 基于机器学习的新闻摘要自动生成系统



\## 项目简介

本项目为课程竞赛设计，旨在“禁用深度学习模型”的限制下，通过特征工程（句子位置、长度、数字特征、TF-IDF 全文余弦相似度）与 LightGBM 分类器构建有监督的抽取式摘要系统。



\## 运行环境

\* Python 3.13 

\* 依赖库：`pip install -r requirements.txt`



\## 运行方法

1\. 请将数据集文件 `train\_dataset.csv` 和 `test\_dataset.csv` 放入项目根目录下。

2\. 在终端运行脚本：

&#x20;  ```bash

&#x20;  python run\_summary\_ml.py




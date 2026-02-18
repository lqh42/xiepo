import os
import numpy as np
import json
from ultralytics import YOLO
    
# 加载模型
model = YOLO("yolov8-seg-CAT-INV-SA-SF.yaml")

# 训练配置
data_path = 'datasets/slide_CUT/slide_cut.yaml'
train_config = {
    'data': data_path,
    'exist_ok': True,
    'batch': 32,  # 减小batch size，降低内存使用
    'boxes': False,  # 禁用检测框显示
    'show_labels': False,  # 禁用标签显示
    'show_conf': False,  # 禁用置信度显示
    'overlap_mask': True,  # 启用mask重叠显示
    'epochs': 200,
    'box': 9.5,
    'cls': 0.5,
    'dfl': 2,
    'pose': 12.0,
    'kobj': 0.5
}

# 训练基础模型（只训练一次）
print("\n=== 开始训练基础模型 ===")
results = model.train(**train_config)


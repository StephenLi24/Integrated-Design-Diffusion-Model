#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2024/7/9 00:01
    @Author : chairc
    @Site   : https://github.com/chairc
"""

# Train
MASTER_ADDR = "localhost"
MASTER_PORT = "12345"
EMA_BETA = 0.995

# Data processing
# Some special parameter settings
# ****** torchvision.transforms.Compose ******
# RandomResizedCrop
RANDOM_RESIZED_CROP_SCALE = (0.8, 1.0)
# Mean in datasets
MEAN = (0.485, 0.456, 0.406)
# Std in datasets
STD = (0.229, 0.224, 0.225)

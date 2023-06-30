#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 19:05
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def device_initializer():
    """
    该函数在程序第一次运行时初始化运行设备信息
    :return: cpu或cuda
    """
    logger.info(msg="Init program, it is checking the basic setting.")
    device_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(device="cuda")
        is_init = torch.cuda.is_initialized()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device=device)
        device_cap = torch.cuda.get_device_capability(device=device)
        device_prop = torch.cuda.get_device_properties(device=device)
        device_dict["is_init"] = is_init
        device_dict["device_count"] = device_count
        device_dict["device_name"] = device_name
        device_dict["device_cap"] = device_cap
        device_dict["device_prop"] = device_prop
        logger.info(msg=device_dict)
    else:
        logger.warning(msg="The device is using cpu.")
        device = torch.device(device="cpu")
    return device
